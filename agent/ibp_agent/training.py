"""Small NumPy policy for numeric features; shared, leakage-safe group split."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .model import INTENTS
from .native import METRICS, BOUNDS
from .parameter_model import DEFAULT_PARAMS
from .settings import atomic_json

PRESERVE = ["color", "brightness", "warmth", "highlights", "details"]
FEATURES = (["before." + k for k in METRICS] + ["present." + k for k in METRICS]
    + [f"hist.{c}.{i}" for c in "YRGB" for i in range(16)]
    + ["intent." + k for k in INTENTS] + ["preserve." + k for k in PRESERVE]
    + ["target." + k for k in METRICS] + ["target_present." + k for k in METRICS])
PARAMS = list(BOUNDS)


def features(payload):
    before = payload["before"]
    targets = {x["metric"]: x["target"] for x in payload.get("targets", [])}
    values = ([before.get(k, 0) for k in METRICS] + [float(k in before) for k in METRICS]
        + [v for c in "YRGB" for v in payload.get("histograms_16", {}).get(c, [0] * 16)]
        + [float(k in payload["intent"]["intents"]) for k in INTENTS]
        + [float(k in payload["intent"].get("preserve", [])) for k in PRESERVE]
        + [targets.get(k, 0) for k in METRICS] + [float(k in targets) for k in METRICS])
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (len(FEATURES),) or not np.isfinite(arr).all():
        raise ValueError("训练特征缺失或无效")
    return arr


def split_rows(rows, minimum=8):
    # Union shoots and identical originals; edits/duplicate uploads can never straddle splits.
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    for row in rows:
        source = row.get("source", row)
        union("group:" + source["group_id"], "sha:" + source.get("sha256", source.get("source_sha256", "")))
    labels = [find("group:" + r.get("source", r)["group_id"]) for r in rows]
    groups = sorted(set(labels), key=lambda x: hashlib.sha256(("ibp-v1:" + x).encode()).hexdigest())
    if len(groups) < minimum:
        raise ValueError(f"至少需要 {minimum} 组不同原图的满意样本；当前 {len(groups)} 组")
    validation = set(groups[:max(2, round(len(groups) * .2))])
    train = [i for i, g in enumerate(labels) if g not in validation]
    valid = [i for i, g in enumerate(labels) if g in validation]
    return train, valid, {"train_groups": [g for g in groups if g not in validation], "validation_groups": sorted(validation),
        "train_ids": [rows[i]["id"] for i in train], "validation_ids": [rows[i]["id"] for i in valid], "seed": 42}


def train_mlp(dataset, progress=lambda x: None, epochs=600):
    rows = [r for r in dataset.records() if r["feedback"]["accepted"] and "policy_input" in r["report"]]
    train_idx, val_idx, split = split_rows(rows)
    x = np.stack([features(r["report"]["policy_input"]) for r in rows])
    bounds = np.asarray([BOUNDS[k] for k in PARAMS], dtype=np.float32)
    low, high = bounds[:, 0], bounds[:, 1]
    raw_y = np.asarray([[r["applied_parameters"].get(k, DEFAULT_PARAMS[k]) for k in PARAMS] for r in rows], dtype=np.float32)
    y = (raw_y - low) / (high - low) * 2 - 1
    mean, scale = x[train_idx].mean(axis=0), x[train_idx].std(axis=0)
    scale = np.maximum(scale, .0001)
    x = np.clip((x - mean) / scale, -10, 10)
    rng = np.random.default_rng(42)
    w = {"w1": (rng.normal(size=(x.shape[1], 64)) / np.sqrt(x.shape[1])).astype(np.float32),
         "b1": np.zeros(64, dtype=np.float32), "w2": (rng.normal(size=(64, len(PARAMS))) * .03).astype(np.float32),
         "b2": np.zeros(len(PARAMS), dtype=np.float32)}
    first, second = {k: np.zeros_like(v) for k, v in w.items()}, {k: np.zeros_like(v) for k, v in w.items()}
    best, best_loss, stale, step = None, float("inf"), 0, 0
    curve = []
    def predict(a):
        return np.tanh(a @ w["w1"] + w["b1"]) @ w["w2"] + w["b2"]
    for epoch in range(epochs):
        for batch in np.array_split(rng.permutation(train_idx), max(1, int(np.ceil(len(train_idx) / 32)))):
            step += 1
            xb, yb = x[batch], y[batch]
            h = np.tanh(xb @ w["w1"] + w["b1"])
            pred = h @ w["w2"] + w["b2"]
            d = 2 * (pred - yb) / pred.size
            dh = (d @ w["w2"].T) * (1 - h * h)
            gradients = {"w2": h.T @ d + .0001 * w["w2"], "b2": d.sum(0),
                         "w1": xb.T @ dh + .0001 * w["w1"], "b1": dh.sum(0)}
            for key in w:
                g = np.clip(gradients[key], -1, 1)
                first[key] = .9 * first[key] + .1 * g
                second[key] = .999 * second[key] + .001 * g * g
                w[key] -= .003 * (first[key] / (1 - .9 ** step)) / (np.sqrt(second[key] / (1 - .999 ** step)) + 1e-8)
        val_loss = float(np.mean((predict(x[val_idx]) - y[val_idx]) ** 2))
        if epoch % 10 == 0:
            curve.append({"epoch": epoch + 1, "validation_mse": val_loss})
            progress({"epoch": epoch + 1, "validation_mse": val_loss})
        if val_loss < best_loss - 1e-6:
            best, best_loss, stale = {k: v.copy() for k, v in w.items()}, val_loss, 0
        else:
            stale += 1
        if stale >= 60:
            break
    w = best
    output = dataset.root / "weights" / ("mlp-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(time.time_ns())[-6:])
    output.mkdir()
    np.savez(output / "weights.npz", **w, mean=mean, scale=scale, low=low, high=high)
    prediction = (np.clip(predict(x[val_idx]), -1, 1) + 1) / 2 * (high - low) + low
    mae = np.mean(np.abs(prediction - raw_y[val_idx]), axis=0)
    baseline = np.mean(np.abs(np.median(raw_y[train_idx], axis=0) - raw_y[val_idx]), axis=0)
    evaluation = {"schema": "ibp-mlp-evaluation-v1", "samples": len(rows), "validation_samples": len(val_idx),
        "split": split, "parameter_mae": dict(zip(PARAMS, map(float, mae))),
        "training_median_baseline_mae": dict(zip(PARAMS, map(float, baseline))), "validation_mse_normalized": best_loss,
        "epochs": epoch + 1, "curve": curve, "scope": "held-out parameter error; not human visual quality or camera validation"}
    contract = {"schema": "ibp-camera-mlp-v1", "features": FEATURES, "parameters": PARAMS,
        "architecture": [len(FEATURES), 64, len(PARAMS)], "activation": "hidden tanh; linear output",
        "preprocessing": "clip((x-mean)/max(scale,0.0001), -10, 10)",
        "output": "(clip(hidden@w2+b2,-1,1)+1)/2*(high-low)+low",
        "input_contract": "identical IBP measurements, 16-bin normalized histograms, structured intents and targets",
        "requires": "camera ISP calibration and renderer-specific parameter translation before deployment",
        "engine": "IBP Core 1.1.0", "weight_layout": "row-major: input x hidden; hidden x output"}
    atomic_json(output / "evaluation.json", evaluation)
    atomic_json(output / "contract.json", contract)
    atomic_json(output / "weights.json", {k: v.tolist() for k, v in dict(w, mean=mean, scale=scale, low=low, high=high).items()})
    return {"path": str(output), "evaluation": evaluation, "weights": str(output / "weights.npz")}


def predict_mlp(weight_file, payload):
    with np.load(weight_file, allow_pickle=False) as w:
        x = np.clip((features(payload) - w["mean"]) / w["scale"], -10, 10)
        y = np.clip(np.tanh(x @ w["w1"] + w["b1"]) @ w["w2"] + w["b2"], -1, 1)
        p = (y + 1) / 2 * (w["high"] - w["low"]) + w["low"]
        return dict(zip(PARAMS, map(float, p)))

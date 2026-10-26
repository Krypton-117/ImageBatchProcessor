"""Small audited ctypes boundary around the upstream Core C ABI."""

import ctypes as C
import math
from pathlib import Path
import numpy as np

METRICS = "mean median p01 p05 p95 p99 rmsContrast entropy logAverage dynamicRangeEV blackClip whiteClip meanSaturation colorfulness meanR meanG meanB labL labA labB sharpness edgeEnergy noiseEstimate".split()


class Host(C.Structure):
    _fields_ = [
        ("width", C.c_uint32),
        ("height", C.c_uint32),
        ("pixels", C.POINTER(C.c_float)),
    ]


class Params(C.Structure):
    _fields_ = [("v", C.c_float * 256)]


class Analysis(C.Structure):
    _fields_ = [
        ("values", C.c_double * 23),
        ("samples", C.c_uint64),
        ("hist", C.c_uint32 * 1024),
    ]


class Target(C.Structure):
    _fields_ = [("params", Params), ("ranges", C.c_double * 8), ("analysis", Analysis)]


# Conservative engineering bounds, not literature-derived aesthetic norms.
BOUNDS = {
    "exposure": (-2.0, 2.0),
    "contrast": (0.7, 1.3),
    "shadows": (-0.6, 0.8),
    "highlights": (-1.0, 0.4),
    "saturation": (0.55, 1.35),
    "vibrance": (-0.15, 0.25),
    "temperature": (-0.6, 0.6),
    "tint": (-0.3, 0.3),
    "denoise": (0, 0.35),
    "sharpen": (0, 0.25),
    "clarity": (-0.15, 0.2),
    "dehaze": (0, 0.15),
    "shoulder": (0, 1),
    "gamma": (0.85, 1.2),
}


def validate_params(values):
    result = {}
    for key, value in values.items():
        if key not in BOUNDS:
            raise ValueError(f"Unsupported parameter: {key}")
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid value: {key}")
        lo, hi = BOUNDS[key]
        if not lo <= value <= hi:
            raise ValueError(f"Out of bounds: {key}")
        result[key] = float(value)
    return result


class Core:
    def __init__(self, bin_dir):
        self.bin_dir = Path(bin_dir).resolve()
        self.lib = C.CDLL(str(self.bin_dir / "ImageBatchProcessorCore.dll"))
        defs = {
            "IBP_LastError": ([], C.c_char_p),
            "IBP_DeviceInfo": ([C.c_void_p], C.c_char_p),
            "IBP_Create": ([C.c_char_p, C.POINTER(C.c_void_p)], C.c_int),
            "IBP_Destroy": ([C.c_void_p], None),
            "IBP_DefaultParams": ([C.POINTER(Params)], None),
            "IBP_DefaultTarget": ([C.POINTER(Target)], None),
            "IBP_LoadImage": ([C.c_char_p, C.c_int, C.POINTER(Host)], C.c_int),
            "IBP_FreeHostImage": ([C.POINTER(Host)], None),
            "UploadImage": (
                [C.c_void_p, C.POINTER(Host), C.POINTER(C.c_uint64)],
                C.c_int,
            ),
            "IBP_ProcessBatch": (
                [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(Params), C.c_uint32],
                C.c_int,
            ),
            "IBP_Analyze": ([C.c_void_p, C.c_uint64, C.POINTER(Analysis)], C.c_int),
            "DownloadImage": ([C.c_void_p, C.c_uint64, C.POINTER(Host)], C.c_int),
            "IBP_ReleaseImage": ([C.c_void_p, C.c_uint64], C.c_int),
            "IBP_SaveImage": ([C.c_char_p, C.POINTER(Host), C.c_int], C.c_int),
            "IBP_SetParam": ([C.POINTER(Params), C.c_char_p, C.c_float], C.c_int),
            "IBP_SaveTarget": ([C.c_char_p, C.POINTER(Target)], C.c_int),
        }
        for name, (args, ret) in defs.items():
            f = getattr(self.lib, name)
            f.argtypes = args
            f.restype = ret
        self.ctx = C.c_void_p()
        self.check(
            self.lib.IBP_Create(
                str(self.bin_dir / "shaders").encode(), C.byref(self.ctx)
            )
        )

    def check(self, status):
        if status:
            raise RuntimeError(
                self.lib.IBP_LastError().decode("utf-8", errors="replace")
            )

    def close(self):
        if self.ctx:
            self.lib.IBP_Destroy(self.ctx)
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def device(self):
        return self.lib.IBP_DeviceInfo(self.ctx).decode()

    def load(self, path, preview=1280):
        h = Host()
        self.check(
            self.lib.IBP_LoadImage(
                str(Path(path).resolve()).encode(), int(bool(preview)), C.byref(h)
            )
        )
        try:
            if h.width * h.height > 100_000_000:
                raise ValueError("Image exceeds 100 MP limit")
            arr = np.ctypeslib.as_array(
                h.pixels, shape=(h.width * h.height * 4,)
            ).reshape(h.height, h.width, 4)
            # Area averaging in linear light, with bounded temporary memory.
            step = max(1, math.ceil(max(h.width, h.height) / preview)) if preview else 1
            if step > 1:
                height = (h.height // step) * step
                width = (h.width // step) * step
                result = (
                    arr[:height, :width]
                    .reshape(height // step, step, width // step, step, 4)
                    .mean(axis=(1, 3))
                )
            else:
                result = arr.copy()
            return np.ascontiguousarray(result, dtype=np.float32)
        finally:
            self.lib.IBP_FreeHostImage(C.byref(h))

    def params(self, values):
        p = Params()
        self.lib.IBP_DefaultParams(C.byref(p))
        for key, value in validate_params(values).items():
            self.check(self.lib.IBP_SetParam(C.byref(p), key.encode(), value))
        return p

    def render(self, original, values, output=None, recipe=None, detailed=False):
        a = np.ascontiguousarray(original, dtype=np.float32)
        h = Host(a.shape[1], a.shape[0], a.ctypes.data_as(C.POINTER(C.c_float)))
        ident = C.c_uint64()
        self.check(self.lib.UploadImage(self.ctx, C.byref(h), C.byref(ident)))
        try:
            p = self.params(values)
            self.check(
                self.lib.IBP_ProcessBatch(self.ctx, C.byref(ident), C.byref(p), 1)
            )
            stats = Analysis()
            self.check(self.lib.IBP_Analyze(self.ctx, ident, C.byref(stats)))
            metrics = dict(zip(METRICS, map(float, stats.values)))
            if not all(math.isfinite(x) for x in metrics.values()):
                raise RuntimeError("Non-finite analysis")
            if output:
                path = Path(output)
                if path.exists():
                    raise FileExistsError(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                decoded = Host()
                self.check(self.lib.DownloadImage(self.ctx, ident, C.byref(decoded)))
                try:
                    self.check(
                        self.lib.IBP_SaveImage(
                            str(path.resolve()).encode(), C.byref(decoded), 95
                        )
                    )
                finally:
                    self.lib.IBP_FreeHostImage(C.byref(decoded))
            if recipe:
                if Path(recipe).exists():
                    raise FileExistsError(recipe)
                t = Target()
                self.lib.IBP_DefaultTarget(C.byref(t))
                t.params = p
                t.analysis = stats
                self.check(
                    self.lib.IBP_SaveTarget(
                        str(Path(recipe).resolve()).encode(), C.byref(t)
                    )
                )
            if detailed:
                hist = list(map(int, stats.hist))
                return {"metrics": metrics, "samples": int(stats.samples),
                        "histograms": {k: hist[i*256:(i+1)*256] for i, k in enumerate(("Y", "R", "G", "B"))}}
            return metrics
        finally:
            self.check(self.lib.IBP_ReleaseImage(self.ctx, ident))

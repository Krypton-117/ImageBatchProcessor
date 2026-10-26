"""Local response-only LoRA SFT on human-approved IBP records (no credentials)."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.training import split_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--local-model", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    train, valid, split = split_rows(rows)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    set_seed(42)
    if args.model not in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"):
        raise ValueError("Unsupported Qwen model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.model != "Qwen/Qwen3-0.6B" and not args.local_model:
        raise ValueError("CPU 训练默认支持 0.6B；较大基座请在 CUDA 机器运行此训练包")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    base = str(args.local_model) if args.local_model else args.model
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    def encode(row):
        messages = row["messages"]
        prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # Build exactly the completion text used at inference, including the chat end token.
        completion = messages[-1]["content"] + tokenizer.eos_token
        before = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        answer = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if len(before) + len(answer) > args.max_length:
            raise ValueError(f"Sample {row['id']} exceeds max-length; nothing was silently truncated")
        return {"input_ids": before + answer, "attention_mask": [1] * (len(before) + len(answer)), "labels": [-100] * len(before) + answer}
    encoded = [encode(row) for row in rows]
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, trust_remote_code=False, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    class Rows(torch.utils.data.Dataset):
        def __init__(self, ids): self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return encoded[self.ids[i]]
    def collate(batch):
        size = max(len(x["input_ids"]) for x in batch)
        return {key: torch.tensor([x[key] + [(-100 if key == "labels" else tokenizer.pad_token_id if key == "input_ids" else 0)] * (size-len(x[key])) for x in batch]) for key in ("input_ids", "attention_mask", "labels")}
    args.output.mkdir(parents=True, exist_ok=False)
    config = TrainingArguments(output_dir=str(args.output / "checkpoints"), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=1e-4, logging_steps=1, save_strategy="no",
        eval_strategy="no", report_to=[], use_cpu=device == "cpu", bf16=dtype == torch.bfloat16,
        dataloader_num_workers=0, seed=42, remove_unused_columns=False)
    class ParameterTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Only materialize vocabulary logits that predict labelled completion tokens.
            # Batch size is deliberately one; prompt hidden states still receive gradients.
            labels = inputs["labels"]
            if labels.shape[0] != 1: raise ValueError("ParameterTrainer requires batch size one")
            positions = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0]
            if not len(positions): raise ValueError("Sample has no completion labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"}, logits_to_keep=positions)
            loss = torch.nn.functional.cross_entropy(outputs.logits[0].float(), labels[0, positions + 1])
            return (loss, outputs) if return_outputs else loss
    trainer = ParameterTrainer(model=model, args=config, train_dataset=Rows(train), eval_dataset=Rows(valid), data_collator=collate)
    trainer.model_accepts_loss_kwargs = False
    baseline = trainer.evaluate()
    result = trainer.train()
    evaluation = trainer.evaluate()
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    report = {"base_model": args.model, "local_model": str(args.local_model.resolve()) if args.local_model else None, "base_revision": getattr(model.config, "_commit_hash", None),
        "base_manifest_sha256": __import__("hashlib").sha256((args.local_model / "manifest.json").read_bytes()).hexdigest() if args.local_model and (args.local_model / "manifest.json").exists() else None,
        "dataset_sha256": __import__("hashlib").sha256(args.data.read_bytes()).hexdigest(), "split": split,
        "baseline": baseline, "evaluation": evaluation, "training": result.metrics, "device": device,
        "loss_scope": "assistant parameter JSON tokens only", "max_length": args.max_length,
        "scope": "held-out token loss; no claim of visual quality or camera compatibility", "completed_at": time.time()}
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "path": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

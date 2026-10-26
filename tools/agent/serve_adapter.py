"""Private loopback OpenAI-compatible endpoint for a locally trained Qwen LoRA."""
import argparse, json, os, secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--base-dir", type=Path)
    args = p.parse_args()
    key = os.environ.pop("IBP_ADAPTER_TOKEN")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    report = json.loads((args.adapter / "evaluation.json").read_text(encoding="utf-8"))
    base = str(args.base_dir) if args.base_dir else report.get("local_model") or report["base_model"]
    if args.base_dir and report.get("base_manifest_sha256"):
        import hashlib
        if hashlib.sha256((args.base_dir / "manifest.json").read_bytes()).hexdigest() != report["base_manifest_sha256"]:
            raise ValueError("Bundled training base differs from adapter provenance")
    revision = report.get("base_revision")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, torch_dtype=torch.float32, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    model.config.use_cache = True
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, code, data):
            content = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        def authorized(self):
            return secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key)
        def do_GET(self):
            self.reply(200 if self.authorized() else 401, {"status": "ready" if self.authorized() else "unauthorized"})
        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {"error": "unauthorized"})
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 131072: raise ValueError("Request too large")
                data = json.loads(self.rfile.read(size))
                text = tokenizer.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False)
                if inputs.input_ids.shape[1] > report.get("max_length", 4096):
                    raise ValueError("Input exceeds trained context length")
                with torch.inference_mode():
                    tokens = model.generate(**inputs, max_new_tokens=min(500, int(data.get("max_tokens", 400))), do_sample=False, pad_token_id=tokenizer.eos_token_id)
                content = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                self.reply(200, {"model": args.adapter.name, "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": inputs.input_ids.shape[1], "completion_tokens": tokens.shape[1]-inputs.input_ids.shape[1]}})
            except Exception as e:
                self.reply(400, {"error": str(e)[:300]})
    print("adapter ready", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

if __name__ == "__main__": main()

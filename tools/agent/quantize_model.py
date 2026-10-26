"""Quantize the hash-verified official Q8 model after download_assets.py."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];folder=root/'.local/models'
spec=json.loads((Path(__file__).with_name('assets.json')).read_text())['model']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=folder/spec['file'];assert sha(source)==spec['sha256']
output=folder/'Qwen3-1.7B-Q4_K_M.gguf'
subprocess.run([str(root/'.local/llama/llama-quantize.exe'),'--allow-requantize',str(source),str(output),'Q4_K_M','4'],check=True)
manifest={'repository':spec['repo'],'revision':spec['revision'],'source_file':spec['file'],'source_sha256':spec['sha256'],'license':spec['license'],'file':output.name,'bytes':output.stat().st_size,'sha256':sha(output),'conversion':'llama.cpp b11889 --allow-requantize Q4_K_M from official Q8_0; additional quantization loss possible'}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

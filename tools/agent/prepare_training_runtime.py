from pathlib import Path
import shutil,subprocess,os
os.environ["PYTHONNOUSERSITE"]="1"
os.environ["PYTHONUTF8"]="1"
root=Path(__file__).resolve().parents[2]
import argparse
p=argparse.ArgumentParser();p.add_argument('--python-base',type=Path,required=True);args=p.parse_args()
base=args.python_base.resolve()
if not (base/'python.exe').exists():raise ValueError('Supply a portable standard CPython 3.12 directory')
dest=root/'.local/training/python'
if not (dest/'python.exe').exists():
 shutil.copytree(base,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','EXTERNALLY-MANAGED'),dirs_exist_ok=True)
py=dest/'python.exe'
subprocess.run([str(py),'-s','-m','ensurepip','--upgrade'],check=True)
subprocess.run([str(py),'-s','-m','pip','install','torch==2.9.0+cpu','--index-url','https://download.pytorch.org/whl/cpu','--disable-pip-version-check'],check=True)
subprocess.run([str(py),'-s','-m','pip','install','transformers==4.57.1','peft==0.17.1','accelerate==1.10.1','numpy==2.3.4','safetensors==0.6.2','--disable-pip-version-check'],check=True)
subprocess.run([str(py),'-s','-c','import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)'],check=True)
(dest.parent/'ready.json').write_text('{"torch":"2.9.0+cpu","transformers":"4.57.1","peft":"0.17.1"}')

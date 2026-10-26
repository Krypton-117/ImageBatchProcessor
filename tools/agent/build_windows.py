"""Build a self-contained community Studio/Agent/inference and CPU-training installer."""
import argparse,hashlib,importlib.metadata,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];LOCAL=ROOT/'.local';BUILD=LOCAL/'agent-build'
DEST=BUILD/'dist/IBPPhotoAgent-0.2.0'

def digest(file):
 with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
 if (Path(sys.base_prefix)/'conda-meta').exists():raise RuntimeError('Use standard CPython 3.12')
 p=argparse.ArgumentParser();p.add_argument('--iscc',type=Path,required=True);args=p.parse_args()
 model_manifest=json.loads((LOCAL/'models/manifest.json').read_text(encoding="utf-8"))
 model=LOCAL/'models'/model_manifest['file'];assert digest(model)==model_manifest['sha256']
 training=LOCAL/'training/python';assert (LOCAL/'training/ready.json').exists()
 training_base=LOCAL/'training/base/Qwen3-0.6B'
 base_manifest=json.loads((training_base/'manifest.json').read_text(encoding="utf-8"))
 for name,item in base_manifest["files"].items():
  assert digest(training_base/name)==item["sha256"],name
 subprocess.run([str(training/'python.exe'),'-I','-c','import torch,transformers,peft;assert "+cpu" in torch.__version__;print(torch.__version__,transformers.__version__,peft.__version__)'],check=True)
 subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--clean','--windowed','--onedir','--name','IBPPhotoAgent-0.2.0','--paths',str(ROOT/'agent'),'--add-data',str(ROOT/'agent/knowledge')+';knowledge','--add-data',str(ROOT/'agent/web')+';web','--collect-all','rawpy','--collect-all','pypdf','--copy-metadata','numpy','--copy-metadata','pillow','--copy-metadata','rawpy','--distpath',str(BUILD/'dist'),'--workpath',str(BUILD/'work'),'--specpath',str(BUILD),str(ROOT/'agent/run_agent.py')],check=True,cwd=ROOT)
 (DEST/'IBPPhotoAgent-0.2.0.exe').rename(DEST/'IBPPhotoAgent.exe')
 for folder in ('app','llama'):shutil.copytree(LOCAL/folder,DEST/folder,dirs_exist_ok=True)
 shutil.copy2(LOCAL/'studio-build/bin/ImageBatchProcessorStudio.exe',DEST/'app/bin/ImageBatchProcessorStudio.exe')
 (DEST/'models').mkdir(exist_ok=True)
 for file in (model.name,'LICENSE','README.md','manifest.json'):shutil.copy2(LOCAL/'models'/file,DEST/'models'/file)
 # Complete isolated CPython + pinned CPU dependencies. No user-site imports or installer-time pip.
 shutil.copytree(training,DEST/'training/python',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','EXTERNALLY-MANAGED'))
 shutil.copy2(LOCAL/'training/ready.json',DEST/'training/ready.json')
 shutil.copytree(training_base,DEST/'training/base/Qwen3-0.6B',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.part','*.ranges'))
 subprocess.run([str(training/'python.exe'),'-I','-m','pip','freeze','--all'],check=True,stdout=(DEST/'training/requirements-lock.txt').open('w',encoding='utf-8'))
 notices=DEST/'licenses';shutil.copytree(BUILD/'licenses',notices,dirs_exist_ok=True)
 shutil.copytree(ROOT/'packaging/licenses',notices/'runtime',dirs_exist_ok=True)
 for name in ('numpy','pillow','rawpy','pypdf','pyinstaller'):
  dist=importlib.metadata.distribution(name)
  for file in dist.files:
   if '/licenses/' in str(file) or str(file).endswith('.dist-info/LICENSE'):
    target=notices/name/Path(str(file).split('.dist-info/')[-1]);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dist.locate_file(file),target)
 for candidate in (Path(sys.base_prefix)/'LICENSE_PYTHON.txt',Path(sys.base_prefix)/'LICENSE.txt',Path(sys.base_prefix)/'LICENSE'):
  if candidate.exists():shutil.copy2(candidate,notices/'Python-LICENSE.txt');break
 for file in ('LICENSE','agent/README.zh-CN.md','agent/THIRD_PARTY_NOTICES.md'):shutil.copy2(ROOT/file,DEST/Path(file).name)
 shutil.copytree(ROOT/'docs/research',DEST/'research',dirs_exist_ok=True,ignore=shutil.ignore_patterns('papers','*.pdf','*.part'))
 for folder in ('agent','Studio','ImageBatchProcessorCore','imgbp','ShaderCompiler','GUI','tools','tests','docs','packaging','external/json','external/LibRaw','external/SDL','external/imgui'):
  shutil.copytree(ROOT/folder,DEST/'sources'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','papers','*.pdf','*.part'))
 for name in ('CMakeLists.txt','README.md','FORMAT.md','QA_RESULTS.md','CHANGELOG.md','LICENSE','THIRD_PARTY_NOTICES.md','natural.imgarg','.gitmodules'):
  if (ROOT/name).exists():shutil.copy2(ROOT/name,DEST/'sources'/name)
 manifest={str(file.relative_to(DEST)):digest(file) for file in DEST.rglob('*') if file.is_file()}
 (DEST/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 subprocess.run([str(args.iscc.resolve()),'/DPayloadDir='+str(DEST),str(ROOT/'packaging/agent.iss')],check=True)
 print('Installer created in',LOCAL/'release',flush=True)

if __name__=='__main__':main()

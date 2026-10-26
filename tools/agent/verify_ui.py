"""Isolated real UI/model/RAW/export checks; labels are synthetic mechanics tests only."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--script',type=Path);p.add_argument('--rag-db',type=Path)
 args=p.parse_args();out=ROOT/'.local'/('ui-check-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir(parents=True)
 env=dict(os.environ,IBP_AGENT_DATA=str(out/'data'),IBP_TRAINING_DATA=str(out/'training'),PYTHONUTF8='1',PYTHONNOUSERSITE='1')
 if args.rag_db:
  (out/'data').mkdir();shutil.copy2(args.rag_db,out/'data/knowledge.sqlite')
 command=[str(args.exe.resolve())]+([str(args.script.resolve())] if args.script else [])+['--no-browser']
 process=subprocess.Popen(command,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 errors=[];evidence={'scope':'real local model and rendering; synthetic review labels; no real teacher or human style validation'}
 try:
  session=out/'data/session.json';end=time.monotonic()+150
  while not session.exists():
   if process.poll() is not None:raise RuntimeError('App exited before startup')
   if time.monotonic()>end:raise TimeoutError('No session')
   time.sleep(.3)
  original_session=json.loads(session.read_text(encoding='utf-8'));url=original_session['url'];base,token=url.split('/#')
  with sync_playwright() as p:
   browser=p.chromium.launch(channel='msedge',headless=True)
   page=browser.new_page(viewport={'width':1600,'height':1000},device_scale_factor=1)
   page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(url);expect(page.locator('#health')).to_contain_text('Qwen3-',timeout=150000)
   request=browser.new_context(extra_http_headers={'X-IBP-Token':token}).request
   def get(route):
    r=request.get(base+route);assert r.ok,r.status;return r.json()
   evidence['status']=get('/api/status')
   page.locator('#settings-open').click();expect(page.locator('#settings-dialog')).to_be_visible()
   assert page.locator('#teacher-key').get_attribute('type')=='password'
   page.locator('[data-close=settings-dialog]').first.click()
   if args.rag_db:
    page.locator('#papers-open').click();page.locator('#rag-query').fill('颗粒太多，干净一点');page.locator('#rag-search').click()
    expect(page.locator('#rag-results .paper-hit').first).to_be_visible(timeout=10000)
    evidence['rag']=get('/api/rag/status');page.locator('[data-close=papers-dialog]').click()
   page.locator('#file').set_input_files(str(ROOT/'.local/raw/canon_40d.CR2'))
   page.locator('#prompt').fill('稍微亮一点，保持自然色彩');page.locator('#run').click(timeout=15000)
   expect(page.locator('#satisfied')).to_be_enabled(timeout=240000)
   assert not page.locator('#error').inner_text()
   assert page.locator('#after').evaluate('(el)=>el.complete && el.naturalWidth>0')
   page.locator('#view-compare').click();page.screenshot(path=str(out/'desktop.png'))
   page.locator('#group-id').fill('TEST-ONLY-UI-MECHANICS')
   page.locator('#satisfied').click();expect(page.locator('#feedback-state')).to_contain_text('已保存 · 满意')
   assert get('/api/dataset')['accepted']==1
   page.locator('#unsatisfied').click();expect(page.locator('#feedback-state')).to_contain_text('已保存 · 不满意')
   assert get('/api/dataset')['accepted']==0
   sample=next((out/'training/samples').glob('*/sample.json'));record=json.loads(sample.read_text(encoding='utf-8'))
   assert record['feedback']['revision']==2 and record['report']['observation']['full_resolution']
   evidence['generation']={'seconds':record['report']['seconds'],'params':record['applied_parameters'],'id':record['id']}
   assert 'api_key' not in sample.read_text(encoding='utf-8')
   page.locator('#export').click();expect(page.locator('#phase')).to_have_text('已导出',timeout=180000)
   assert list((out/'data/jobs'/record['id']/'exports').glob('*.tif*'))
   page.locator('#training-open').click();expect(page.locator('#count-rejected')).to_have_text('1')
   page.locator('#prepare-qwen').click();expect(page.locator('#training-state')).to_contain_text('已完成',timeout=120000)
   evidence['training_environment']=get('/api/training')
   page.locator('#train-mlp').click();expect(page.locator('#training-state')).to_contain_text('至少需要 8 组',timeout=10000)
   page.locator('[data-close=training-dialog]').click()
   page.set_viewport_size({'width':900,'height':700});page.screenshot(path=str(out/'compact.png'))
   assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
   page.set_viewport_size({'width':480,'height':850});page.screenshot(path=str(out/'mobile.png'))
   assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
   page.set_viewport_size({'width':1600,'height':1000});page.reload();expect(page.locator('#history .thumbnail').nth(1)).to_be_visible()
   # Second launcher must reuse the same authenticated service and exit.
   second=subprocess.run(command,env=env,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   assert second.returncode==0 and json.loads(session.read_text(encoding='utf-8'))['pid']==original_session['pid']
   evidence['launcher_reuse']=True;evidence['browser_errors']=errors;assert not errors,errors
   request.post(base+'/api/quit',data={})
   browser.close()
  evidence['ok']=True
 finally:
  try:
   import urllib.request
   cleanup=json.loads((out/'data/session.json').read_text(encoding='utf-8'));cleanup_base,cleanup_token=cleanup['url'].split('/#')
   req=urllib.request.Request(cleanup_base+'/api/quit',b'{}',{'X-IBP-Token':cleanup_token,'Content-Type':'application/json'})
   urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=5).read()
  except Exception:pass
  if process.poll() is None:
   try:process.wait(15)
   except subprocess.TimeoutExpired:process.terminate()
  (out/'result.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
  print(out,flush=True)

if __name__=='__main__':main()

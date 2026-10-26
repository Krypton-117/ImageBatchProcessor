import copy, json, sys, tempfile, threading, unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'agent'))
from ibp_agent.settings import Settings, endpoint
from ibp_agent.dataset import Dataset, sha256
from ibp_agent.parameter_model import ParameterModel, DEFAULT_PARAMS
from ibp_agent.training import split_rows, train_mlp, predict_mlp, FEATURES
from ibp_agent.native import BOUNDS

class TrainingWorkflowTests(unittest.TestCase):
 def test_keys_encrypted_and_update_preserves_or_clears(self):
  with tempfile.TemporaryDirectory() as folder:
   s=Settings(folder);s.update({'teacher':{'api_key':'fixture-secret-DO-NOT-SHIP'}})
   self.assertNotIn('fixture-secret',Path(folder,'settings.json').read_text())
   self.assertNotIn('fixture-secret',json.dumps(s.public()))
   s=Settings(folder);self.assertEqual(s.profile('teacher')['api_key'],'fixture-secret-DO-NOT-SHIP')
   s.update({'teacher':{'api_key':''}});self.assertTrue(s.public()['teacher']['has_key'])
   s.update({'teacher':{'clear_key':True}});self.assertFalse(Settings(folder).public()['teacher']['has_key'])
 def test_endpoint_rejects_leaking_credentials(self):
  for url in ['http://remote.example','https://user:pw@remote.example','https://remote.example?key=secret','file:///x']:
   with self.assertRaises(ValueError):endpoint(url)
  self.assertEqual(endpoint('http://127.0.0.1:8765/'),'http://127.0.0.1:8765')
 def test_annotation_revision_rebuilds_exports(self):
  with tempfile.TemporaryDirectory() as folder:
   base=Path(folder);d=Dataset(base/'data');original=base/'original.dng';original.write_bytes(b'fixture-original')
   render=base/'render';render.mkdir()
   for name in ['before.jpg','after.jpg','preview.imgarg']:(render/name).write_bytes(name.encode())
   report={'source_sha256':sha256(original),'source_name':'fixture.dng','params':{'exposure':.2},'metric_contract':{},'prediction':{'messages':[{'role':'user','content':'fixture'}]}}
   d.label('a'*32,report,original,render,True)
   row=json.loads((d.root/'exports/qwen-sft.jsonl').read_text())
   self.assertEqual(json.loads(row['messages'][-1]['content'])['params']['exposure'],.2)
   result=d.label('a'*32,report,original,render,False)
   self.assertEqual(result['revision'],2);self.assertEqual(d.status()['rejected'],1)
   self.assertEqual((d.root/'exports/accepted.jsonl').read_text(),'')
   self.assertTrue((d.root/'samples'/('a'*32)/'history/sample-1.json').exists())
   with self.assertRaises(ValueError):d.label('b'*32,report,original,render,'yes')
 def test_identical_originals_and_shoots_do_not_leak(self):
  rows=[{'id':str(i),'group_id':'g'+str(i),'source_sha256':str(i)}for i in range(12)]
  rows += [{'id':'dup','group_id':'another','source_sha256':'0'},{'id':'same-shoot','group_id':'g1','source_sha256':'x'}]
  train,val,_=split_rows(rows)
  self.assertEqual(0 in train,12 in train);self.assertEqual(1 in train,13 in train)
  self.assertFalse(set(train)&set(val))
  with self.assertRaises(ValueError):split_rows(rows[:2])
 def test_mlp_roundtrip_on_isolated_synthetic_samples(self):
  with tempfile.TemporaryDirectory() as folder:
   d=Dataset(folder)
   for i in range(12):
    item=d.root/'samples'/str(i);item.mkdir()
    payload={'before':{'mean':i/12,'median':i/12},'intent':{'intents':['brighten'],'preserve':[]},'targets':[{'metric':'median','target':.5}]}
    row={'id':str(i),'source':{'sha256':str(i),'group_id':str(i)},'feedback':{'accepted':True},'report':{'policy_input':payload},'applied_parameters':dict(DEFAULT_PARAMS,exposure=(.5-i/12))}
    (item/'sample.json').write_text(json.dumps(row))
   result=train_mlp(d,epochs=35);weights=Path(result['weights'])
   self.assertTrue(weights.exists());pred=predict_mlp(weights,payload)
   for k,v in pred.items():self.assertTrue(BOUNDS[k][0]-1e-6<=v<=BOUNDS[k][1]+1e-6)
   contract=json.loads((weights.parent/'contract.json').read_text());self.assertEqual(contract['architecture'][0],len(FEATURES))
   self.assertFalse(set(result['evaluation']['split']['train_ids']) & set(result['evaluation']['split']['validation_ids']))
 def test_openai_parameter_contract_and_secret_exclusion(self):
  received={}
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_POST(self):
    received.update(path=self.path,auth=self.headers.get('Authorization'),body=json.loads(self.rfile.read(int(self.headers['Content-Length']))))
    data=json.dumps({'choices':[{'message':{'content':'{"params":{"exposure":0.3}}'}}],'model':'fixture','usage':{}}).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   model=ParameterModel({'base_url':f'http://127.0.0.1:{server.server_port}','api_key':'fixture-key','model':'teacher'})
   prediction=model.propose({'request':'test'});self.assertEqual(prediction['params']['exposure'],.3)
   self.assertEqual(received['path'],'/v1/chat/completions');self.assertEqual(received['auth'],'Bearer fixture-key')
   self.assertNotIn('fixture-key',json.dumps(prediction))
  finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main()

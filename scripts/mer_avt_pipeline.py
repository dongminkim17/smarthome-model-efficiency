#!/usr/bin/env python3
"""Resumable MER AVT SFT -> LM pruning/KD -> encoder pruning/feature-KD workflow.

The lightweight coordinator submits immutable phase DAGs to the existing Suzy
queue. It never owns a GPU or selects sources using official test performance.
"""
from __future__ import annotations
import argparse,copy,json,os,shlex,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint,exclusive_lock,same_process,now
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.data import load_records
from smarthome_efficiency.training import load_config,config_fingerprint
from smarthome_efficiency.mer_transcripts import validate_input_manifest,SHA256
from project_queue import validate_submission
from project_queue_jobs import entry,validate_export
from queue_ffn_expansion import training_chain
from launch_pruning_followup import checkpoint_identity,pin_best
FAMILY='mer_avt_20261023'
TEACHER=FAMILY+'_teacher';LM=FAMILY+'_lm';ENCODER=FAMILY+'_encoder'
DATA_ROOT=(ROOT/'data_mer_avt_chinese').resolve()
ARMS=(
 ('BALANCED_FEATURE_SPLITLR','BALANCED',16,1e-5),
 ('BALANCED_FEATURE_R64','BALANCED',64,1e-5),
 ('VISION_FEATURE','VISION',16,1e-4),
)
ARCHITECTURES={'BALANCED':(4352,3328,2048),'VISION':(4352,3072,2176)}
PYTHON=sys.executable

def source_config(root=ROOT,data_root=DATA_ROOT):
 root=Path(root);data_root=Path(data_root)
 protocol=validate_input_manifest(data_root/'MER2023','trainval')
 if not protocol or protocol['language']!='chinese' or protocol['source']['sha256']!=SHA256:raise ValueError('Verified pinned Chinese AVT data required')
 cfg=load_config(root/'configs/mer6_avt.yaml');cfg.update(data_root=str(data_root.resolve()),use_transcript=True,
  cache_dir=str(root/'artifacts/pilot_cache_mer_avt_chinese'),epochs=5,seed=42,split_seed=42,
  micro_batch_size=2,gradient_accumulation_steps=2,gradient_checkpointing=True,lora_rank=16,lora_alpha=32,
  lora_dropout=0.,lora_target='qv',learning_rate=1e-4,objective='sft',teacher_checkpoint=None,teacher_cache=None,
  eval_modes=['logit','greedy'],selection_mode='logit',eval_every_fraction=.5)
 return cfg

def prepare_id():return TEACHER+'/prepare:mer6_avt'
def stress_path(state):return str(Path(state['queue_root'])/'stress_cases/mer6_avt_chinese.json')
def train_id(family,arm):return family+'/train:mer6_'+arm.lower()
def chain(family,state,root,arm,cfg,source,deps,**metadata):
 manifest=Path(root)/'artifacts/full'/family/'workflow.json'
 jobs=training_chain(family,manifest,state['queue_root'],root,'mer6',arm,cfg,source,deps)
 for j in jobs:
  j['payload'].update(input_protocol='audio+video+author-provided Chinese transcript',**metadata)
  if j['kind']=='smoke_train':j['payload']['memory_cases']=stress_path(state)
  if j['kind']=='smoke_test':j['payload']['reload_latest']=True
  if j['kind'] in ('smoke_train','recovery_train'):
   j['priority']=60 if j['kind']=='smoke_train' else 20
   j['payload']['estimated_seconds']=360 if j['kind']=='smoke_train' else 21600
 return jobs

def finish_spec(state,family,jobs,root,description):
 manifest=Path(root)/'artifacts/full'/family/'workflow.json'
 document={'format_version':1,'status':'waiting','phase':'per_job_dependencies','description':description,
  'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'checkpoint_selection_uses_test':False,
  'input_protocol':'audio+video+author-provided Chinese transcript','existing_source_selections_unchanged':True}
 for j in jobs:document[j['location']['field']].append(copy.deepcopy(j['payload']))
 spec={'schema_version':1,'jobs':jobs,'documents':{str(manifest):document}};validate_submission(state,spec);return spec

def located(family,kind,payload,root,deps=(),priority=30,resource='gpu'):
 return entry(family+'/'+payload['id'],kind,payload,{'manifest':str(Path(root)/'artifacts/full'/family/'workflow.json'),
  'field':'preparation_jobs','key_field':'id','key':payload['id']},list(deps),priority=priority,resource=resource)

def build_teacher(state,root=ROOT,data_root=DATA_ROOT):
 cfg=source_config(root,data_root)
 prep=located(TEACHER,'prepare',{'id':'prepare:mer6_avt','task':'mer6','output':stress_path(state),'config':cfg},root,priority=100,resource='cpu')
 jobs=[prep]+chain(TEACHER,state,root,'SFT_QV16',cfg,None,[prep['id']],initialization='fresh q/v rank16 LoRA on original Qwen2.5-Omni-3B; AVT; no AV teacher reuse')
 return finish_spec(state,TEACHER,jobs,root,'MER AVT original teacher: fresh five-epoch q/v rank16 SFT, private stress/reload gate and official best-validation test')

def completed_source(state,key):
 j=next(j for j in state['jobs'] if j['id']==key)
 if j['status']!='completed':raise ValueError('Source training must finish: '+key)
 checkpoint=pin_best(j['payload']['run_dir'])
 if not checkpoint:raise ValueError('No validation-selected checkpoint')
 return j,Path(checkpoint)

def export(family,state,root,source,method,widths,deps,calibration_name):
 cfg=read_json(source/'metadata.json')['config'];output=Path(root)/'artifacts/compact'/family/calibration_name
 request={'method':method,'source_checkpoint':str(source),'source_adapter_sha256':sha256_file(source/'adapter/adapter_model.safetensors'),
  'source_metadata_sha256':sha256_file(source/'metadata.json'),'keep_layers':None,'ffn_width':widths[0],
  'calibration_samples':128,'calibration_seed':42,'calibration_cache':str(Path(root)/'artifacts/compact'/family/'calibration'/('mer6_'+('encoders' if method=='encoder_ffn_activation' else 'lm')+'_n128.json'))}
 counts=expected_counts(*widths)
 payload={'id':'export:'+calibration_name,'task':'mer6','output':str(output),'request':request,'source_identity':checkpoint_identity(source),
  'generate_calibration':True,'calibration_sha256':None,'expected_language_parameters':counts['language'],'estimated_seconds':1800}
 if method=='encoder_ffn_activation':
  request.update(budget_scope='separate',audio_ffn_width=widths[1],vision_ffn_width=widths[2])
  payload.update(expected_parameters=counts,expected_combined_encoder_parameters=counts['audio_encoder']+counts['vision_encoder'],
   source_base_manifest_sha256=sha256_file(Path(cfg['model'])/'compression.json'))
 return located(family,'prune_export',payload,root,deps)

def build_lm(state,root=ROOT):
 teacher_job,source=completed_source(state,train_id(TEACHER,'SFT_QV16'));cfg=copy.deepcopy(read_json(source/'metadata.json')['config'])
 if cfg['task']!='mer6' or not cfg['use_transcript'] or cfg['objective']!='sft':raise ValueError('AVT SFT teacher required')
 cache=Path(root)/'artifacts/cache'/FAMILY
 identity=checkpoint_identity(source)
 logits=located(LM,'followup',{'id':'teacher:mer6_avt','kind':'teacher','name':'mer6_avt','task':'mer6',
  'output':str(cache/'logits'),'teacher_checkpoint':str(source),'limit':None,'source_identity':identity},root,[teacher_job['id']])
 features=located(LM,'encoder_cache',{'id':'encoder_cache:mer6_avt','task':'mer6','output':str(cache/'encoder_features'),
  'teacher_checkpoint':str(source),'source_identity':identity,'expected_examples':2698},root,[teacher_job['id']])
 compact=export(LM,state,root,source,'ffn_activation',(4352,5120,3420),[teacher_job['id']],'mer6_ffn4352')
 cfg.update(model=compact['payload']['output'],lora_target='decoder_linear',objective='sft_kd',teacher_checkpoint=str(source),
  teacher_cache=str(cache/'logits/cache.json'),distillation_alpha=.75,distillation_temperature=2.)
 jobs=[logits,features,compact]+chain(LM,state,root,'KD075',cfg,str(source),[compact['id'],logits['id'],prepare_id()],
  initialization='AVT SFT teacher merged then activation-FFN4352 pruning; fresh decoder-linear rank16 class-KD recovery')
 return finish_spec(state,LM,jobs,root,'MER AVT: fresh training-only class/feature targets, LM activation FFN4352 export and five-epoch KD075 recovery')

def build_encoder(state,root=ROOT):
 teacher_job,teacher=completed_source(state,train_id(TEACHER,'SFT_QV16'));lm_job,source=completed_source(state,train_id(LM,'KD075'))
 by={j['id']:j for j in state['jobs']};cache_id=LM+'/encoder_cache:mer6_avt';logit_id=LM+'/teacher:mer6_avt'
 if any(by[k]['status']!='completed' for k in (cache_id,logit_id)):raise ValueError('AVT teacher caches must finish')
 cache=Path(root)/'artifacts/cache'/FAMILY;feature_path=cache/'encoder_features/cache.json'
 cfg=copy.deepcopy(read_json(teacher/'metadata.json')['config'])
 jobs=[];exports={}
 for name,widths in ARCHITECTURES.items():
  e=export(ENCODER,state,root,source,'encoder_ffn_activation',widths,[lm_job['id']],name.lower());jobs.append(e);exports[name]=e
 for arm,arch,rank,encoder_lr in ARMS:
  c=copy.deepcopy(cfg);c.update(model=exports[arch]['payload']['output'],lora_rank=rank,lora_alpha=2*rank,
   lora_target='decoder_encoder_ffn',objective='sft_kd',teacher_checkpoint=str(teacher),teacher_cache=str(cache/'logits/cache.json'),
   feature_teacher_cache=str(cache/'encoder_features'),feature_audio_weight=.1,feature_vision_weight=.1,
   distillation_alpha=.75,distillation_temperature=2.,encoder_learning_rate=encoder_lr)
  added=chain(ENCODER,state,root,arm,c,str(source),[exports[arch]['id'],cache_id,logit_id,prepare_id()],
   architecture=arch,budget_scope='separate',expected_parameters=expected_counts(*ARCHITECTURES[arch]),
   initialization='AVT LM KD075 winner merged, encoder-only activation pruning, fresh decoder+encoder-FFN LoRA; fixed original AVT teacher')
  for j in added:
   if j['kind'] in ('smoke_train','recovery_train'):j['payload']['expected_feature_cache_sha256']=sha256_file(feature_path)
  jobs.extend(added)
 return finish_spec(state,ENCODER,jobs,root,'MER AVT: balanced rank16/rank64 splitLR and vision-allocation feature-KD recovery; three independent five-epoch runs and official tests')

def submit(state,spec,family):
 q=Path(state['queue_root']);saved=q/'submissions'/(family+'.json')
 if saved.exists():
  existing=read_json(saved)
  if fingerprint(existing)!=fingerprint(spec):raise ValueError('Existing phase submission differs')
 else:atomic_json(saved,spec)
 path=q/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec)
 return str(path)

def tick(state,root=ROOT,data_root=DATA_ROOT):
 by={j['id']:j for j in state['jobs']}
 scoped=[j for j in state['jobs'] if j['id'].startswith((TEACHER+'/',LM+'/',ENCODER+'/'))]
 if any(j['status'] in ('failed','blocked') for j in scoped):return ('failed',None,None)
 if not any(k.startswith(TEACHER+'/') for k in by):
  return ('teacher',build_teacher(state,root,data_root),TEACHER)
 if by[train_id(TEACHER,'SFT_QV16')]['status']!='completed':return ('waiting_teacher',None,None)
 if not any(k.startswith(LM+'/') for k in by):return ('lm',build_lm(state,root),LM)
 if by[train_id(LM,'KD075')]['status']!='completed' or by[LM+'/encoder_cache:mer6_avt']['status']!='completed':return ('waiting_lm_or_features',None,None)
 if not any(k.startswith(ENCODER+'/') for k in by):return ('encoder',build_encoder(state,root),ENCODER)
 terminal=[j for j in state['jobs'] if j['id'].startswith((TEACHER+'/',LM+'/',ENCODER+'/'))]
 if any(j['status'] in ('failed','blocked') for j in terminal):return ('failed',None,None)
 return ('completed' if all(j['status']=='completed' for j in terminal) else 'waiting_final_tests',None,None)

def ensure_controller(q,reload=False):
 if socket.gethostname().split('.')[0]!='suzy':raise ValueError('Run the pipeline coordinator on Suzy')
 s=read_json(q/'state.json');active=[(j.get('pid'),j.get('identity')) for j in s['jobs'] if j['status']=='running' and j.get('pid')]
 if reload and same_process(s.get('scheduler_pid'),s.get('scheduler_identity')):
  subprocess.run([PYTHON,'-B',str(ROOT/'scripts/project_queue.py'),'stop'],check=True)
  end=time.time()+30
  while same_process(s.get('scheduler_pid'),s.get('scheduler_identity')):
   if time.time()>end:raise RuntimeError('Controller detach timeout')
   time.sleep(.25)
  # Workers may finish naturally during this short handoff. The replacement
  # controller validates their durable result instead of treating exit as loss.
  atomic_json(ROOT/'artifacts/implementation'/FAMILY/'controller_handoff.json',{'at':now(),'worker_pids_before':[pid for pid,_ in active],
   'still_live':[pid for pid,identity in active if same_process(pid,identity)],'workers_signalled':False})
 if not same_process(s.get('scheduler_pid'),s.get('scheduler_identity')):
  session='smarthome_project_queue'
  if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise RuntimeError('Unexpected controller session')
  (q/'dispatch.enabled').write_text('Dispatch enabled for user-approved AVT pipeline '+now()+'\n')
  command=shlex.join(['env','TMPDIR=/tmp','PYTHONDONTWRITEBYTECODE=1','PATH='+str(Path(PYTHON).parent)+':'+os.environ.get('PATH',''),PYTHON,'-B','-u',str(ROOT/'scripts/project_queue.py'),'run'])
  subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'exec '+command+' >> '+shlex.quote(str(q/'scheduler.log'))+' 2>&1'],check=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run','status']);p.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy');args=p.parse_args()
 folder=ROOT/'artifacts/full'/FAMILY
 if args.action=='status':
  path=folder/'pipeline.json';print(json.dumps(read_json(path) if path.exists() else {'status':'not_started'},indent=2));return 0
 if args.action=='plan':
  state=read_json(args.queue_root/'state.json');spec=build_teacher(state);atomic_json(ROOT/'artifacts/implementation'/FAMILY/'teacher_plan.json',spec)
  print('Validated five scientific training runs in three phases: teacherSFT, LM-KD075, three encoder recoveries. Full data, five epochs, automatic tests; no submission made.');return 0
 folder.mkdir(parents=True,exist_ok=True)
 with exclusive_lock(folder/'.coordinator.lock'):
  ensure_controller(args.queue_root,reload=True)
  while True:
   state=read_json(args.queue_root/'state.json');phase,spec,family=tick(state)
   if spec is not None:submit(state,spec,family)
   status={'updated_at':now(),'phase':phase,'input_protocol':'audio+video+Chinese transcript','queue_root':str(args.queue_root),
    'families':[TEACHER,LM,ENCODER],'selection_uses_test':False}
   atomic_json(folder/'pipeline.json',status);print(json.dumps(status),flush=True)
   if phase=='completed':return 0
   if phase=='failed':return 1
   time.sleep(30)
if __name__=='__main__':raise SystemExit(main())

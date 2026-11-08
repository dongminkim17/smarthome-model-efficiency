"""Approved 18-run matrix, pinned exports/calibration and independent execution gates."""
import copy,sys,tempfile,unittest
from pathlib import Path
import torch,yaml
from transformers import Qwen2_5OmniThinkerConfig,Qwen2_5OmniThinkerForConditionalGeneration
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from queue_full_model_b_followups import build_submission,FAMILY,TASKS,B_CONTROLS,NEW_BUDGETS,B_WIDTHS
from project_queue_jobs import entry,native_command
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.pruning import sha256_file,parameter_counts
from smarthome_efficiency.queueing import atomic_json,validate_graph
from smarthome_efficiency.training import DEFAULTS,load_config


class BFollowupTests(unittest.TestCase):
    def fixture(self,root):
        jobs=[]
        for task in TASKS:
            source=root/task/'source';base=root/task/'base';model=root/task/'B';cache=root/task/'cache.json';cal=root/task/'calibration.json'
            atomic_json(base/'compression.json',{'fixture':'originalFFN4352'})
            (source/'adapter').mkdir(parents=True);(source/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
            cfg=dict(DEFAULTS,task=task,model=str(model),lora_target='decoder_encoder_ffn',objective='sft_kd',
                     teacher_checkpoint=str(root/task/'teacher'),teacher_cache=str(cache),distillation_alpha=.75,
                     micro_batch_size=2,gradient_accumulation_steps=2)
            atomic_json(source/'metadata.json',{'config':cfg|{'model':str(base)}})
            atomic_json(cache,{'status':'completed','spec':{'teacher_checkpoint':cfg['teacher_checkpoint'],'split':'train','test_set_used':False}})
            atomic_json(cal,{'fixture':'training-only calibration'})
            side={'task':task,'method':'multimodal_ffn_activation','budget_scope':'total','parameters_after':expected_counts(*B_WIDTHS),
                  'verification':{'passed':True},'source_checkpoint':str(source),'source_adapter_sha256':sha256_file(source/'adapter/adapter_model.safetensors'),
                  'source_metadata_sha256':sha256_file(source/'metadata.json'),'calibration_samples':128,'calibration_seed':42,
                  'calibration_cache':str(cal),'calibration':{'sha256':sha256_file(cal)}}
            atomic_json(model/'compression.json',side);name=task+'_b_lm1280_a1536_v1024'
            jobs.extend([entry('full_model_2b_20261020/export:'+name,'prune_export',{'task':task,'status':'completed','output':str(model)}),
                entry('full_model_2b_20261020/train:'+name,'recovery_train',{'task':task,'status':'completed','config':cfg}),
                entry('calibrated_pruning_20261019/teacher:'+task,'followup',{'task':task,'status':'completed'}),
                entry('prepare:'+task,'prepare',{'task':task,'status':'completed'},resource='cpu')])
        return {'jobs':jobs,'documents':{},'queue_root':str(root/'queue')}

    def test_exact_matrix_and_controls_preserve_sources_and_previous_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);original=copy.deepcopy(state);spec=build_submission(state,root)
            self.assertEqual(state,original);validate_graph(state['jobs']+spec['jobs']);self.assertEqual(len(spec['jobs']),78)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];self.assertEqual(len(trains),18)
            self.assertEqual(sum(j['kind']=='prune_export' for j in spec['jobs']),6)
            by={j['id']:j for j in spec['jobs']}
            for task in TASKS:
                cfgs={j['payload']['arm']:j['payload']['config'] for j in trains if j['task']==task}
                ref=next(j['payload']['config'] for j in state['jobs'] if j['id']=='full_model_2b_20261020/train:'+task+'_b_lm1280_a1536_v1024')
                self.assertEqual(set(cfgs),{a for a,_ in B_CONTROLS}|{a for a,_ in NEW_BUDGETS})
                for arm,changes in B_CONTROLS:
                    self.assertEqual(cfgs[arm],ref|changes)
            for j in trains:
                p=j['payload'];cfg=p['config'];self.assertIsNone(p['init_adapter']);self.assertEqual(cfg['epochs'],5)
                self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4)
                self.assertLess(p['expected_parameters']['total'],2e9)
                reload=by[j['deps'][0]];smoke=by[reload['deps'][0]];self.assertEqual(smoke['kind'],'smoke_train')
                self.assertEqual(by[FAMILY+'/test:'+p['name']]['deps'],[j['id']])
                path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(path),cfg)
            for j in spec['jobs']:
                if j['kind']=='prune_export':
                    self.assertFalse(j['payload']['generate_calibration']);self.assertIsNotNone(j['payload']['calibration_sha256'])
                    command,_=native_command(state,j,root);self.assertIn('--reuse',command)
            (root/'mintrec2/calibration.json').write_text('changed')
            with self.assertRaises(ValueError):build_submission(state,root)

    def test_new_production_counts_match_native_meta_models(self):
        source=ROOT/'artifacts/compact/calibrated_pruning_20261019/mintrec2_ffn_activation'
        if not source.exists():self.skipTest('Production config unavailable')
        totals=[1976071424,1973982464]
        for (arm,widths),total in zip(NEW_BUDGETS,totals):
            cfg=Qwen2_5OmniThinkerConfig.from_pretrained(source,local_files_only=True)
            cfg.text_config.intermediate_size=widths[0];cfg.audio_config.encoder_ffn_dim=widths[1]
            cfg.vision_config.intermediate_size=widths[2];cfg._attn_implementation='sdpa'
            with torch.device('meta'):model=Qwen2_5OmniThinkerForConditionalGeneration(cfg)
            self.assertEqual(parameter_counts(model),expected_counts(*widths),arm);self.assertEqual(parameter_counts(model)['total'],total)


if __name__=='__main__':unittest.main()

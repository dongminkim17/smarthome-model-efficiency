"""Distillation gradients, cache provenance, compatibility, and graph ordering."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from smarthome_efficiency.distillation import TeacherLogitCache,distillation_kl,sample_fingerprint
from smarthome_efficiency.training import DEFAULTS,load_config,validate_resume
from launch_pruning_followup import build_jobs,refresh_jobs,pin_best,completed_result
import test_evaluation


class DistillationTests(unittest.TestCase):
    def test_kl_direction_temperature_scaling_and_frozen_teacher(self):
        student=torch.tensor([[1.,-1.,.5],[2.,0.,-1.]],requires_grad=True)
        teacher=torch.tensor([[0.,2.,-1.],[1.,3.,0.]],requires_grad=True)
        temperature=2.
        loss=distillation_kl(student,teacher,temperature);loss.backward()
        expected=temperature/student.shape[0]*(torch.softmax(student.detach()/temperature,-1)-torch.softmax(teacher.detach()/temperature,-1))
        torch.testing.assert_close(student.grad,expected,atol=1e-6,rtol=1e-5)
        self.assertIsNone(teacher.grad)
        self.assertLess(abs(float(distillation_kl(teacher.detach(),teacher.detach(),temperature))),1e-6)

    def test_microbatch_kd_gradient_matches_single_example_accumulation(self):
        value=torch.randn(4,6,requires_grad=True);teacher=torch.randn(4,6)
        distillation_kl(value,teacher,2.).backward();expected=value.grad.clone();value.grad=None
        for i in range(4):(distillation_kl(value[i:i+1],teacher[i:i+1],2.)/4).backward()
        torch.testing.assert_close(value.grad,expected)
        with self.assertRaises(ValueError):distillation_kl(value,teacher[:2],2.)

    def test_model_exposes_causal_live_class_logits_for_kd(self):
        fixture=test_evaluation.ModelEvaluationTests();fixture.setUp();runner=fixture.runner
        result=runner.loss_batch(fixture.batch,0,return_class_gradients=True)
        self.assertTrue(result['class_logits_for_kd'].requires_grad)
        self.assertFalse(result['class_logits'].requires_grad)
        other=runner.loss_batch(fixture.batch,1,return_class_gradients=True)
        torch.testing.assert_close(result['class_logits_for_kd'],other['class_logits_for_kd'])
        loss=distillation_kl(result['class_logits_for_kd'],torch.tensor([[3.,-2.]]),2.)
        loss.backward();self.assertNotEqual(float(runner.model.weight.grad),0.)

    def test_cache_rejects_changed_media_teacher_order_or_nontrain_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);media=root/'clip.mp4';media.write_bytes(b'test')
            record={'id':'clip','video':str(media),'label':'a'}
            spec={'split':'train','test_set_used':False,'labels':['a','b'],'teacher':'fixed'}
            payload={'status':'completed','spec':spec,'rows':{'clip':{'sample_sha256':sample_fingerprint(record),'logits':[1.,2.]}}}
            path=root/'cache.json';path.write_text(json.dumps(payload))
            cache=TeacherLogitCache(path,spec,[record]);self.assertEqual(cache.batch([record],'cpu').tolist(),[[1.,2.]])
            with self.assertRaises(ValueError):TeacherLogitCache(path,spec|{'labels':['b','a']},[record])
            with self.assertRaises(ValueError):TeacherLogitCache(path,spec|{'teacher':'changed'},[record])
            media.write_bytes(b'changed')
            with self.assertRaises(ValueError):TeacherLogitCache(path,spec,[record])
            payload['spec']=spec|{'split':'test','test_set_used':True};path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):TeacherLogitCache(path,payload['spec'],[])

    def test_legacy_sft_resume_and_distillation_config_guards(self):
        old={k:v for k,v in DEFAULTS.items() if k not in ['objective','teacher_checkpoint','teacher_cache','distillation_alpha','distillation_temperature']}
        validate_resume(old,DEFAULTS,1,1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cfg.yaml'
            path.write_text(yaml.safe_dump({'model':'model','objective':'sft_kd'}))
            with self.assertRaises(ValueError):load_config(path)
            path.write_text(yaml.safe_dump({'model':'model','objective':'sft_kd','teacher_checkpoint':'teacher','teacher_cache':'cache'}))
            cfg=load_config(path);self.assertEqual(cfg['distillation_temperature'],2.)
            with self.assertRaises(ValueError):validate_resume(DEFAULTS,cfg,1,1)


class WorkflowTests(unittest.TestCase):
    def source(self):
        return {'jobs':[{'name':f'{task}_{method}','task':task,'source_checkpoint':f'/teacher/{task}',
                         'compact_model':f'/compact/{task}/{method}','output':f'/sft/{task}/{method}',
                         'config':dict(DEFAULTS,task=task,model=f'/compact/{task}/{method}'),
                         'status':'running','gpu':i+1}
                        for i,task in enumerate(['mer6','mintrec2','mintrec20']) for method in ['depth_uniform','ffn_magnitude']]}

    def test_graph_has_three_shared_caches_and_fresh_initialization(self):
        source=self.source();jobs=build_jobs(source,Path('/followup'))
        self.assertEqual(len(jobs),21)
        self.assertEqual(sum(j['kind']=='teacher' for j in jobs),3)
        for job in jobs:
            if job['kind']=='kd_train':
                cfg=job['config'];self.assertEqual(cfg['objective'],'sft_kd');self.assertEqual(cfg['epochs'],5)
                self.assertEqual(cfg['model'],job['compact_model'])
                self.assertIn(f'sft_test:{job["name"]}',job['deps'])
                self.assertIn(f'teacher:{job["task"]}',job['deps'])
                self.assertNotIn('init_adapter',cfg)
        smoke=build_jobs(source,Path('/smoke'),smoke=True)
        for job in smoke:
            if job['kind'].endswith('test'):self.assertEqual(job['split'],'dev');self.assertEqual(job['limit'],6)

    def test_kd_waits_for_test_and_cache_and_does_not_read_test_scores(self):
        source=self.source();jobs=build_jobs(source,Path('/followup'));by_id={j['id']:j for j in jobs}
        selected=by_id['kd_train:mer6_depth_uniform'];original=copy.deepcopy(selected['config'])
        refresh_jobs(jobs,source,False);self.assertEqual(selected['status'],'waiting')
        by_id['sft_test:mer6_depth_uniform'].update(status='completed',result={'logit':{'accuracy':0.}})
        refresh_jobs(jobs,source,False);self.assertEqual(selected['status'],'waiting')
        by_id['teacher:mer6']['status']='completed'
        refresh_jobs(jobs,source,False);self.assertEqual(selected['status'],'ready');self.assertEqual(selected['config'],original)
        by_id['sft_test:mer6_depth_uniform']['result']['logit']['accuracy']=1.
        self.assertEqual(selected['config'],original)

    def test_failed_dependency_blocks_only_its_branch(self):
        source=self.source();jobs=build_jobs(source,Path('/followup'))
        jobs[0]['status']='failed'
        refresh_jobs(jobs,source,False)
        self.assertEqual(jobs[1]['status'],'blocked');self.assertEqual(jobs[2]['status'],'blocked')
        self.assertTrue(any(j['status']=='ready' and j['kind']=='teacher' for j in jobs))

    def test_best_checkpoint_is_pinned_only_after_training_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);cp=run/'checkpoints/step-00000002';(cp/'adapter').mkdir(parents=True)
            (cp/'metadata.json').write_text(json.dumps({'global_step':2}))
            (cp/'adapter/adapter_config.json').write_text('{}');(cp/'adapter/adapter_model.safetensors').touch()
            (cp.parent/'best').symlink_to(cp.name)
            report={'status':'running','test_set_used':False,'best_step':2};(run/'report.json').write_text(json.dumps(report))
            self.assertIsNone(pin_best(run))
            report['status']='completed';(run/'report.json').write_text(json.dumps(report))
            self.assertEqual(pin_best(run),str(cp))
            report['best_step']=1;(run/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):pin_best(run)

    def test_full_evaluation_rejects_partial_test_results(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            job={'kind':'sft_test','run_dir':str(run),'split':'test','checkpoint':'fixed','limit':None}
            report={'status':'completed','split':'test','checkpoint':'fixed','eval_modes':['logit','greedy'],
                    'test_set_used':True,'partial_split':True,'evaluated_count':6,'full_split_count':445}
            (run/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):completed_result(job,False)


if __name__=='__main__':unittest.main()

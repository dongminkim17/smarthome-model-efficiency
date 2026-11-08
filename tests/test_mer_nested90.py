"""Nested holdout isolation, safe cache reuse, and the twelve-run time budget."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.cache_extension import partition_spec, seed_extension, pin_generated_caches
from smarthome_efficiency.data import LABELS, load_records
from smarthome_efficiency.distillation import sample_fingerprint
from smarthome_efficiency.training import DEFAULTS, load_config, validate_resume
from queue_mer_nested90 import definitions, PARTITION


class NestedHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def make_mer(self):
        root=self.root/'MER2023';(root/'manifests').mkdir(parents=True)
        records=[]
        for label in LABELS['mer6']:
            for i in range(40):
                path=root/f'{label}_{i}.avi';path.touch()
                records.append({'id':path.stem,'video':path.name,'label':label})
        (root/'manifests/trainval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_retained_holdout_is_never_previously_trained_and_test_is_unread(self):
        self.make_mer()
        old=[load_records('mer6',s,data_root=self.root) for s in ('train','dev')]
        new=[load_records('mer6',s,data_root=self.root,mer_partition=PARTITION) for s in ('train','dev')]
        ids=lambda rows:{r['id'] for r in rows}
        self.assertEqual(tuple(map(len,new)),(216,24))
        self.assertLessEqual(ids(old[0]),ids(new[0]))
        self.assertLessEqual(ids(new[1]),ids(old[1]))
        self.assertFalse(ids(new[0])&ids(new[1]))
        self.assertEqual(ids(new[0])|ids(new[1]),ids(old[0])|ids(old[1]))
        self.assertEqual({label:sum(r['label']==label for r in new[1]) for label in LABELS['mer6']},
                         {label:4 for label in LABELS['mer6']})
        self.assertEqual(new[1],load_records('mer6','dev',data_root=self.root,mer_partition=PARTITION))
        with self.assertRaises(FileNotFoundError):load_records('mer6','test',data_root=self.root,mer_partition=PARTITION)
        with self.assertRaises(ValueError):load_records('mer6','dev',data_root=self.root,mer_partition=PARTITION,mer_dev_fraction=.1)
        with self.assertRaises(ValueError):load_records('mintrec2','train',data_root=self.root,mer_partition=PARTITION)

    def test_optional_config_does_not_change_legacy_but_full_resume_rejects_split_change(self):
        import yaml
        path=self.root/'cfg.yaml';cfg=DEFAULTS|{'task':'mer6','model':str(self.root)}
        path.write_text(yaml.safe_dump(cfg))
        old=load_config(path);self.assertNotIn('mer_partition',old)
        path.write_text(yaml.safe_dump(cfg|{'mer_partition':PARTITION}))
        new=load_config(path);self.assertEqual(new['mer_partition'],PARTITION)
        with self.assertRaisesRegex(ValueError,'mer_partition'):validate_resume(old,new,1,1)
        path.write_text(yaml.safe_dump(cfg|{'task':'mintrec2','mer_partition':PARTITION}))
        with self.assertRaises(ValueError):load_config(path)

    def make_cache(self):
        records=[]
        for i in range(3):
            path=self.root/f'clip{i}.avi';path.write_bytes(b'media')
            records.append({'id':str(i),'video':str(path),'label':'neutral','transcript':None})
        spec={'split':'train','test_set_used':False,'labels':['a','b'],'teacher_checkpoint':'unchanged'}
        old={'status':'completed','num_examples':2,'spec':spec,
             'rows':{r['id']:{'sample_sha256':sample_fingerprint(r),'logits':[1.,2.]} for r in records[:2]}}
        source=self.root/'original.json';source.write_text(json.dumps(old))
        return records,old,source

    def test_extension_reuses_valid_rows_in_new_directory_without_changing_source(self):
        records,old,source=self.make_cache();before=source.read_bytes();target=self.root/'new'
        result=seed_extension(source,target,records,PARTITION)
        self.assertEqual((result['reused_rows'],result['additional_rows']),(2,1))
        self.assertEqual(source.read_bytes(),before)
        request=json.loads((target/'request.json').read_text())
        self.assertEqual(request['spec']['training_partition'],PARTITION)
        self.assertEqual(set(request['record_fingerprints']),{'0','1','2'})
        progress=(target/'progress.jsonl').read_text()
        seed_extension(source,target,records,PARTITION)
        self.assertEqual((target/'progress.jsonl').read_text(),progress)
        altered=copy.deepcopy(records);altered[0]['label']='angry'
        with self.assertRaises(ValueError):seed_extension(source,target,altered,PARTITION)

    def test_extension_rejects_heldout_rows_and_test_provenance(self):
        records,old,source=self.make_cache()
        with self.assertRaisesRegex(ValueError,'outside'):seed_extension(source,self.root/'bad',records[1:],PARTITION)
        self.assertFalse((self.root/'bad').exists())
        old['spec']['test_set_used']=True;source.write_text(json.dumps(old))
        with self.assertRaises(ValueError):seed_extension(source,self.root/'badtest',records,PARTITION)
        with self.assertRaises(ValueError):seed_extension(source,source,records,PARTITION)

    def test_generated_cache_hashes_pin_once_and_then_reject_mutation(self):
        records,old,_=self.make_cache()
        classpath=self.root/'class.json';featuredir=self.root/'features';featuredir.mkdir()
        old['spec']=partition_spec(old['spec'],PARTITION)
        classpath.write_text(json.dumps(old))
        feature={'status':'completed','num_examples':2,'rows':old['rows'],'spec':{'teacher':old['spec']}}
        (featuredir/'cache.json').write_text(json.dumps(feature))
        payload={'config':{'teacher_cache':str(classpath),'feature_teacher_cache':str(featuredir),
                           'teacher_checkpoint':'unchanged','mer_partition':PARTITION},'expected_cache_examples':2}
        pin_generated_caches(payload);before=copy.deepcopy(payload)
        pin_generated_caches(payload);self.assertEqual(before,payload)
        old['rows']['0']['logits']=[5.,6.];classpath.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError,'changed'):pin_generated_caches(payload)

    def test_twelve_configs_keep_geometry_fixed_and_schedule_long_jobs_first(self):
        arms=definitions()
        self.assertEqual(len(arms),12)
        self.assertEqual(sum(a[2] for a in arms),4)
        self.assertTrue(all(not a[2] and a[3]['epochs']==5 for a in arms[:8]))
        self.assertTrue(all(a[2] and a[3]['epochs']==3 for a in arms[8:]))
        self.assertGreater(min(a[4] for a in arms[:8]),max(a[4] for a in arms[8:]))
        self.assertEqual({a[1] for a in arms},{16,64})
        self.assertEqual({a[3]['distillation_alpha'] for a in arms},{.5,.75})
        allowed={'epochs','learning_rate','encoder_learning_rate','distillation_alpha'}
        self.assertTrue(all(set(a[3])<=allowed for a in arms))
        self.assertEqual(sum(a[3]['epochs'] for a in arms),52)


if __name__=='__main__':unittest.main()

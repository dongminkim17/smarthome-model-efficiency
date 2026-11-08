"""Transcript joins, unchanged AV protocol and isolation from test labels/caches."""
import csv,hashlib,inspect,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from smarthome_efficiency.mer_transcripts import create_view,parse_transcripts,validate_input_manifest
from smarthome_efficiency.data import LABELS,load_records
from smarthome_efficiency.model import build_prompt,build_input_prompt
from smarthome_efficiency.distillation import sample_fingerprint

class MerTranscriptTests(unittest.TestCase):
 def fixture(self,root):
  src=root/'original';(src/'manifests').mkdir(parents=True);(src/'raw').mkdir();(src/'labels').mkdir()
  rows=[]
  for label in LABELS['mer6']:
   for i in range(10):
    key=f'{label}_{i}';(src/'raw'/(key+'.avi')).touch();rows.append({'id':key,'label':label,'video':'raw/'+key+'.avi'})
  test=[{'id':'heldout','label':'happy','video':'raw/heldout.avi'}];(src/'raw/heldout.avi').touch()
  for name,rr in [('trainval',rows),('semi_test',test)]:
   (src/'manifests'/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rr))
  transcript=root/'text.csv';stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['name','chinese','english'])
  for i,r in enumerate(rows+test):writer.writerow([r['id'],'中文内容' if i else '','English speech' if i else ''])
  transcript.write_text(stream.getvalue());view=root/'avt'
  return src,transcript,view

 def test_view_preserves_ids_labels_splits_and_every_clip_including_empty_text(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src,text,view=self.fixture(root);before={p.name:p.read_bytes() for p in (src/'manifests').iterdir()}
   protocol=create_view(src,view,text,expected_counts={'trainval':60,'semi_test':1})
   for name,body in before.items():self.assertEqual((src/'manifests'/name).read_bytes(),body)
   for split in ['train','dev','test']:
    # Original root layout is deliberately wrapped under MER2023 just as production.
    original=root/'original_wrapper';original.mkdir(exist_ok=True)
    if not (original/'MER2023').exists():(original/'MER2023').symlink_to(src)
    a=load_records('mer6',split,data_root=original);b=load_records('mer6',split,data_root=view)
    self.assertEqual([{k:v for k,v in x.items() if k!='transcript'} for x in a],[{k:v for k,v in x.items() if k!='transcript'} for x in b])
    self.assertTrue(all(x['transcript'] is None for x in a));self.assertTrue(all(isinstance(x['transcript'],str) for x in b))
   self.assertEqual(len(protocol['coverage']['trainval']['empty_ids']),1)
   self.assertEqual(create_view(src,view,text,expected_counts={'trainval':60,'semi_test':1}),protocol)
   # Training reads the trainval view only: the official test manifest may be absent.
   (view/'MER2023/manifests/semi_test.jsonl').unlink()
   self.assertEqual(len(load_records('mer6','train',data_root=view)),48)

 def test_missing_duplicate_and_tampered_text_are_rejected(self):
  with self.assertRaises(ValueError):parse_transcripts(b'name,chinese,english\nx,a,b\nx,c,d\n')
  for change in ('missing','text','manifest'):
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);src,text,view=self.fixture(root)
    if change=='missing':
     text.write_text('name,chinese,english\nunknown,a,b\n')
     with self.assertRaises(ValueError):create_view(src,view,text,expected_counts={'trainval':60,'semi_test':1})
    else:
     create_view(src,view,text,expected_counts={'trainval':60,'semi_test':1})
     path=text if change=='text' else view/'MER2023/manifests/trainval.jsonl'
     with path.open('a') as f:f.write('changed')
     with self.assertRaises(ValueError):load_records('mer6','train',data_root=view)

 def test_prompt_is_label_blind_and_av_intent_strings_are_unchanged(self):
  labels=LABELS['mer6'];record={'id':'clip','label':'sad','transcript':'中文的说话内容'}
  prompt=build_input_prompt(record,labels,'mer6');self.assertIn(record['transcript'],prompt)
  self.assertEqual(prompt,build_input_prompt(dict(record,label='happy'),labels,'mer6'))
  self.assertNotIn(record['transcript'],build_input_prompt(dict(record,transcript=None),labels,'mer6'))
  self.assertIn('[No transcript available]',build_input_prompt(dict(record,transcript=''),labels,'mer6'))
  self.assertEqual(build_prompt(dict(record,transcript=None),labels,'mer6'),build_input_prompt(dict(record,transcript=None),labels,'mer6'))
  self.assertEqual(build_prompt(record,LABELS['mintrec2'],'mintrec2'),build_input_prompt(record,LABELS['mintrec2'],'mintrec2'))
  # Frozen historical prompt source hash protects existing teacher-cache identities.
  source_hash=hashlib.sha256(inspect.getsource(build_prompt).strip().encode()).hexdigest()
  self.assertEqual(source_hash,'109f93ae8ee6e2ea883576809db28160c710b36099141e6e2d256ce1638e1000')

 def test_transcripts_change_record_fingerprint(self):
  with tempfile.TemporaryDirectory() as d:
   video=Path(d)/'video';video.touch();r={'id':'clip','video':str(video),'label':'happy','transcript':None}
   self.assertNotEqual(sample_fingerprint(r),sample_fingerprint(dict(r,transcript='speech')))

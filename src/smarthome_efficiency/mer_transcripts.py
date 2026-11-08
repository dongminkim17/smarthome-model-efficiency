"""Pinned MER-author transcripts and separate immutable AVT dataset views."""
import csv,hashlib,io,json
from pathlib import Path
REVISION='89120b780ab18b28349f0e37dcc12b8cc253cb02'
RELATIVE='MERBench/dataset/mer2023-dataset-process/transcription-engchi-polish.csv'
URL='https://raw.githubusercontent.com/zeroQiaoba/MERTools/'+REVISION+'/'+RELATIVE
SHA256='8f773eab800f8f3080a7551f25b5bfb17e3e78cd39742414256f32901bb0be97'
BLOB_SHA1='2e15175318907e86a9549f79f6cbd9887831d07d'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parse_transcripts(body,language='chinese'):
 if language not in ('chinese','english'):raise ValueError('Choose Chinese or English author transcripts')
 reader=csv.DictReader(io.StringIO(body.decode('utf-8-sig')))
 if reader.fieldnames!=['name','chinese','english']:raise ValueError('Unexpected author transcript columns')
 result={}
 for row in reader:
  key=row['name']
  if not key or key in result or row.get(language) is None:raise ValueError('Duplicate/malformed transcript ID')
  value=row[language].strip()
  # Empty entries explicitly supplied by the authors remain empty; never drop
  # their clips or synthesize text from emotion annotations.
  result[key]='' if value.lower() in ('nan','none','null') else value
 return result

def create_view(source_root,output_root,transcript_path,language='chinese',expected_counts=None):
 source_root=Path(source_root).resolve();output_root=Path(output_root).resolve();transcript_path=Path(transcript_path).resolve()
 if output_root==source_root or source_root in output_root.parents:raise ValueError('Use a separate dataset root outside the original MER directory')
 text=parse_transcripts(transcript_path.read_bytes(),language);counts=expected_counts or {'trainval':3373,'semi_test':834}
 manifests={};coverage={};seen=set()
 for split,count in counts.items():
  rows=[json.loads(x) for x in (source_root/'manifests'/(split+'.jsonl')).read_text().splitlines() if x.strip()]
  if len(rows)!=count:raise ValueError('Unexpected official split size: '+split)
  joined=[];empty=[]
  for row in rows:
   key=row['id']
   if key in seen:raise ValueError('Duplicate ID or split overlap')
   seen.add(key)
   if key not in text:raise ValueError('Missing transcript record: '+key)
   if not (source_root/row['video']).is_file():raise FileNotFoundError(row['video'])
   if text[key]=='':empty.append(key)
   joined.append(dict(row,transcript=text[key]))
  manifests[split]=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in joined)
  coverage[split]={'rows':len(rows),'nonempty':len(rows)-len(empty),'empty_ids':empty,'source_manifest_sha256':digest(source_root/'manifests'/(split+'.jsonl'))}
 target=output_root/'MER2023';target.mkdir(parents=True,exist_ok=True);(target/'manifests').mkdir(exist_ok=True)
 for name in ('raw','labels'):
  link=target/name
  if link.exists() or link.is_symlink():
   if link.resolve()!=(source_root/name).resolve():raise ValueError('Dataset link points to a different source')
  else:link.symlink_to(source_root/name,target_is_directory=True)
 for name,content in manifests.items():
  path=target/'manifests'/(name+'.jsonl')
  if path.exists() and path.read_text()!=content:raise ValueError('Refusing to overwrite a different transcript view')
  if not path.exists():path.write_text(content)
 protocol={'format_version':1,'input_protocol':'audio+video+author-provided transcript','language':language,
  'source':{'repository':'https://github.com/zeroQiaoba/MERTools','revision':REVISION,'path':RELATIVE,'url':URL,
    'transcript_path':str(transcript_path),'sha256':digest(transcript_path),'author_rows':len(text)},
  'coverage':coverage,'manifest_sha256':{name:digest(target/'manifests'/(name+'.jsonl')) for name in manifests},
  'missing_text_policy':'Keep author-empty transcripts as empty strings; retain all clips; prompt marks text unavailable. No emotion labels/descriptions used as text.'}
 path=target/'manifests/input_protocol.json'
 if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Refusing to replace existing input provenance')
 if not path.exists():path.write_text(json.dumps(protocol,indent=2)+'\n')
 return protocol

def validate_input_manifest(root,name):
 root=Path(root);path=root/'manifests/input_protocol.json'
 if not path.exists():return None
 p=json.loads(path.read_text())
 if p.get('format_version')!=1 or p.get('language') not in ('chinese','english'):raise ValueError('Unsupported MER transcript protocol')
 if digest(p['source']['transcript_path'])!=p['source']['sha256']:raise ValueError('Author transcript file changed')
 if digest(root/'manifests'/(name+'.jsonl'))!=p['manifest_sha256'][name]:raise ValueError('Transcript manifest changed')
 return p

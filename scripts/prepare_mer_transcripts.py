#!/usr/bin/env python3
"""Download the pinned MER-author CSV and create a separate AVT data root."""
import argparse,hashlib,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from smarthome_efficiency.mer_transcripts import REVISION,URL,SHA256,BLOB_SHA1,create_view,validate_input_manifest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,default=ROOT/'data/MER2023');p.add_argument('--output-root',type=Path,default=ROOT/'data_mer_avt_chinese');p.add_argument('--language',choices=['chinese','english'],default='chinese');p.add_argument('--verify-only',action='store_true');args=p.parse_args()
 folder=args.source_root/'transcripts'/('merbench_'+REVISION[:12]);path=folder/'transcription-engchi-polish.csv'
 if not path.exists():
  if args.verify_only:raise FileNotFoundError(path)
  body=urllib.request.urlopen(URL,timeout=60).read()
  if hashlib.sha256(body).hexdigest()!=SHA256 or hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()!=BLOB_SHA1:raise ValueError('Author transcript checksum mismatch')
  folder.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:raise ValueError('Pinned transcript file changed')
 if args.verify_only:
  protocol=validate_input_manifest(args.output_root/'MER2023','trainval')
  if not protocol:raise FileNotFoundError('No prepared AVT view')
  validate_input_manifest(args.output_root/'MER2023','semi_test')
 else:protocol=create_view(args.source_root,args.output_root,path,args.language)
 link=ROOT/('data_mer_avt_'+args.language)
 if link.exists() or link.is_symlink():
  if link.resolve()!=args.output_root.resolve():raise ValueError('Project transcript symlink points elsewhere')
 elif args.verify_only:raise FileNotFoundError(link)
 else:link.symlink_to(args.output_root.resolve(),target_is_directory=True)
 print(json.dumps({'data_root':str(args.output_root),'project_link':str(link),'language':args.language,'coverage':{k:{x:y for x,y in v.items() if x!='empty_ids'} for k,v in protocol['coverage'].items()},'sha256':protocol['source']['sha256']},indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())

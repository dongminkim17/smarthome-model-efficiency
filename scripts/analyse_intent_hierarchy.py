#!/usr/bin/env python3
"""Training-free validation diagnostic of existing <2B complete-model 20-class runs.

Read only saved dev predictions of each original 20-class validation winner.
Never read official test rows, fit a threshold, or change checkpoint selection.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from smarthome_efficiency.hierarchical_validation import project_validation_rows
from smarthome_efficiency.queueing import atomic_json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyse(index_path, output):
    index=json.loads(Path(index_path).read_text());results=[];skipped=[]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for record in index['runs']:
        if (record['task']!='mintrec20' or record['training_status']!='completed'
                or not record.get('total_parameters') or record['total_parameters']>=2_000_000_000):continue
        run=Path(record['run_dir']);report=json.loads((run/'report.json').read_text())
        checkpoint=(run/'checkpoints/best').resolve()
        metadata=json.loads((checkpoint/'metadata.json').read_text())
        step=metadata['global_step'];predictions=run/'predictions'/f'dev-step-{step:08d}.jsonl'
        if not predictions.is_file():
            skipped.append({'id':record['id'],'reason':'Original best-checkpoint dev predictions unavailable'});continue
        manifest=json.loads((run/'data_manifest.json').read_text())
        provenance=json.loads((run/'provenance.json').read_text())
        if (report['status']!='completed' or report['test_set_used'] is not False
                or metadata['config']['task']!='mintrec20' or report['best_step']!=step
                or len(manifest['dev'])!=445 or manifest['full_dev_count']!=445):
            raise ValueError('Expected completed full-validation 20-class checkpoint: '+record['id'])
        rows=[json.loads(line) for line in predictions.read_text().splitlines()]
        fine_accuracy=sum(r['true_label']==r['logit_predicted_label'] for r in rows)/len(rows)
        if abs(fine_accuracy-report['best_score'])>1e-10:
            raise ValueError('Saved dev predictions differ from the original selected validation metric')
        projected,metrics=project_validation_rows(rows,manifest['dev'],provenance['labels'])
        name=record['family']+'__'+record['name'];path=output/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in projected))
        result={'id':record['id'],'family':record['family'],'name':record['name'],'arm':record['arm'],
                'total_parameters':record['total_parameters'],'checkpoint':str(checkpoint),'checkpoint_step':step,
                'checkpoint_selection':'original logit 20-class validation accuracy; unchanged',
                'fine_validation_accuracy':fine_accuracy,'coarse_validation':metrics,
                'prediction_source':str(predictions),'prediction_sha256':digest(predictions),
                'manifest_sha256':digest(run/'data_manifest.json'),
                'metadata_sha256':digest(checkpoint/'metadata.json'),'provenance_sha256':digest(run/'provenance.json'),
                'projected_predictions':str(path.resolve()),'split':'dev','test_set_used':False}
        results.append(result)
    report={'status':'completed','created_at':datetime.now(timezone.utc).isoformat(),'split':'dev','test_set_used':False,
            'rule':'sum twenty-way softmax probabilities at T=1 within official coarse categories; fixed argmax',
            'threshold_fitted':False,'checkpoint_selection_changed':False,'additional_parameters':0,
            'source_index':str(Path(index_path).resolve()),'source_index_generated_at':index.get('generated_at_kst'),
            'results':results,'skipped':skipped}
    atomic_json(output/'report.json',report)
    with (output/'results.csv').open('w',newline='') as stream:
        fields=['family','name','arm','total_parameters','fine_validation_accuracy','coarse_probability_sum_accuracy','coarse_hard_map_accuracy']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in results:
            writer.writerow({k:r[k] for k in fields[:5]}|{'coarse_probability_sum_accuracy':r['coarse_validation']['probability_sum']['accuracy'],
                         'coarse_hard_map_accuracy':r['coarse_validation']['hard_map']['accuracy']})
    print('Completed validation-only projection for '+str(len(results))+' whole-model checkpoints: '+str(output/'report.json'))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=ROOT/'experiments/runs.json')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();analyse(args.index,args.output)


if __name__=='__main__':main()

"""Extend immutable train-only teacher caches without recomputing old targets."""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path

from .distillation import sample_fingerprint
from .pruning import sha256_file
from .queueing import atomic_json, read_json


def partition_spec(old_spec, partition, *, features=False):
    result = copy.deepcopy(old_spec)
    teacher = result['teacher'] if features else result
    if teacher.get('training_partition') not in (None, partition):
        raise ValueError('Cannot extend a cache from a different custom partition')
    if teacher['split'] != 'train' or teacher['test_set_used'] is not False:
        raise ValueError('Only train-only teacher targets may be reused')
    teacher['training_partition'] = partition
    return result


def seed_extension(source, output, records, partition, *, features=False):
    """Reuse old rows only if EVERY source ID belongs to the new training set.

    Feature files are hard-linked, never edited. New extraction writes only
    missing IDs into the new directory; historical cache manifests stay intact.
    The regular producer verifies request, hashes, shapes and resumable rows.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Cache extension needs a new immutable output directory')
    old_path = source/'cache.json' if features else source
    old = read_json(old_path)
    if old.get('status') != 'completed' or len(old['rows']) != old['num_examples']:
        raise ValueError('Source teacher cache is incomplete')
    fingerprints = {r['id']: sample_fingerprint(r) for r in records}
    if len(fingerprints) != len(records) or not set(old['rows']) <= set(fingerprints):
        raise ValueError('Source cache includes IDs outside the NEW training set')
    spec = partition_spec(old['spec'], partition, features=features)
    request = {'spec':spec, 'record_fingerprints':fingerprints}
    output.mkdir(parents=True, exist_ok=True)
    request_path = output/'request.json'
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError('Cannot change an existing extension request')
    if not request_path.exists(): atomic_json(request_path, request)
    if (output/'cache.json').exists():
        final = read_json(output/'cache.json')
        if (final.get('status') != 'completed' or final['spec'] != spec
                or set(final['rows']) != set(fingerprints)
                or any(row['sample_sha256'] != fingerprints[key] for key,row in final['rows'].items())):
            raise ValueError('Existing extension is incomplete or stale')
        return {'reused_rows':len(old['rows']), 'additional_rows':len(records)-len(old['rows'])}
    rows = []
    for key,row in old['rows'].items():
        if row['sample_sha256'] != fingerprints[key]:
            raise ValueError('Cannot reuse stale teacher targets: '+key)
        if features:
            from .feature_distillation import feature_filename
            if row['file'] != feature_filename(key): raise ValueError('Invalid feature filename')
            src = source/'features'/row['file']; dst = output/'features'/row['file']
            if sha256_file(src) != row['sha256']: raise ValueError('Corrupt source encoder features')
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists(): os.link(src, dst)
            if sha256_file(dst) != row['sha256']: raise ValueError('Changed linked encoder features')
        elif (len(row['logits']) != len(spec['labels'])
              or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in row['logits'])):
            raise ValueError('Invalid reusable class logits')
        rows.append({'id':key, **row})
    progress = output/'progress.jsonl'
    if not progress.exists():
        # A single durable atomic replacement makes interruption/retry safe.
        temporary = progress.with_suffix('.jsonl.tmp')
        with temporary.open('w') as stream:
            for row in rows: stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(progress)
    provenance = {'source':str(source), 'source_cache_sha256':sha256_file(old_path),
                  'reused_rows':len(rows), 'additional_rows':len(records)-len(rows),
                  'partition':partition, 'test_set_used':False}
    atomic_json(output/'extension.json', provenance)
    return provenance


def pin_generated_caches(payload):
    """Pin newly produced caches at first dispatch, then reject later mutation."""
    cfg = payload['config']; class_path = Path(cfg['teacher_cache'])
    feature_path = Path(cfg['feature_teacher_cache'])/'cache.json'
    teacher, features = read_json(class_path), read_json(feature_path)
    partition = cfg['mer_partition']; expected = payload['expected_cache_examples']
    if (teacher.get('status') != 'completed' or features.get('status') != 'completed'
            or teacher['spec'].get('training_partition') != partition
            or features['spec']['teacher'] != teacher['spec']
            or teacher['spec']['teacher_checkpoint'] != cfg['teacher_checkpoint']
            or teacher['spec']['split'] != 'train' or teacher['spec']['test_set_used'] is not False
            or teacher['num_examples'] != expected or features['num_examples'] != expected
            or len(teacher['rows']) != expected or set(teacher['rows']) != set(features['rows'])):
        raise ValueError('New training-only caches do not match the requested partition/teacher')
    for field,path in (('expected_teacher_cache_sha256',class_path),('expected_feature_cache_sha256',feature_path)):
        value = sha256_file(path)
        if payload.get(field) not in (None, value):
            raise ValueError('Pinned generated cache changed: '+field)
        payload[field] = value

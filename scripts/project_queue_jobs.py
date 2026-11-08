"""Adapters from existing experiment manifests to the persistent project DAG."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from launch_experiments import ROOT, PROJECT_ENV, stamp
from launch_pruning_sweep import SOURCES
from launch_pruning_followup import build_jobs, completed_result, checkpoint_identity, pin_best
from launch_recovery_sweep import build_command, validate_result
from recovery_experiments import TASKS, select_sources, materialize_jobs
from smarthome_efficiency.queueing import SCHEMA_VERSION, fingerprint, now, process_identity, validate_graph, read_json
from smarthome_efficiency.training import DEFAULTS, config_fingerprint

NAIVE = ROOT / 'artifacts/full/naive_pruning_20261019'
CALIBRATED = ROOT / 'artifacts/full/calibrated_pruning_20261019'
RANK = ROOT / 'artifacts/full/recovery_ablation_20261019'
CONTROLS = ROOT / 'artifacts/full/recovery_controls_20261019'
CACHE = NAIVE / 'followup/teachers'


def read(path):
    return read_json(path)


def entry(key, kind, payload, location=None, deps=(), resource='gpu', priority=0):
    status = payload.get('status', 'waiting')
    status = status if status in ('completed', 'failed', 'blocked', 'running') else 'waiting'
    job = {'id': key, 'kind': kind, 'task': payload.get('task'), 'payload': copy.deepcopy(payload),
           'location': location, 'deps': list(deps), 'resource': resource, 'status': status,
           'priority': priority, 'attempts': [], 'created_at': now(), 'retry_count': 0}
    if status == 'running':
        identity = process_identity(payload['pid'])
        if identity is not None and identity['command'] != payload['command']:
            raise ValueError('Imported worker command differs from manifest')
        job.update(imported=True, pid=payload['pid'], identity=identity,
                   gpu=payload.get('gpu'), gpu_uuid=payload.get('gpu_uuid'), log=payload.get('log'),
                   exit_status='unknown_for_imported_worker')
    return job


def import_project(queue_root):
    """Read-only shadow import. No worker signals or artifact writes."""
    jobs = []
    documents = {}
    source_ids = {task: [] for task in TASKS}
    source_entries = {task: [] for task in TASKS}
    for base in (NAIVE, CALIBRATED):
        path = base / 'sweep.json'
        source = read(path)
        documents[str(path)] = source
        for original in source['jobs']:
            key = f'{base.name}/sft:{original["name"]}'
            location = {'manifest': str(path), 'field': 'jobs', 'key': original['name'], 'key_field': 'name'}
            job = entry(key, 'sft', original, location)
            jobs.append(job)
            source_ids[original['task']].append(key)
            source_entries[original['task']].append(original)
        follow_path = base / 'followup/workflow.json'
        if follow_path.exists():
            follow = read(follow_path)
        else:
            follow = {'format_version': 1, 'source_sweep': str(base), 'mode': 'test-and-distill',
                      'smoke': False, 'epochs': 5, 'alpha': .5, 'temperature': 2.,
                      'status': 'running', 'jobs': build_jobs(source, follow_path.parent, teacher_cache_root=CACHE)}
        documents[str(follow_path)] = follow
        for original in follow['jobs']:
            key = f'{base.name}/{original["id"]}'
            deps = [f'{base.name}/{value}' for value in original['deps']]
            if original['kind'] == 'sft_test':
                deps.append(f'{base.name}/sft:{original["name"]}')
            location = {'manifest': str(follow_path), 'field': 'jobs', 'key': original['id'], 'key_field': 'id'}
            job = entry(key, 'followup', original, location, deps, priority=40 if original['kind'].endswith('test') else 20)
            if original['kind'] == 'teacher' and (Path(original['output']) / 'cache.json').exists():
                job.update(status='completed', result=completed_result(original, False))
                job['payload'].update(status='completed', result=job['result'])
            jobs.append(job)
    for task in TASKS:
        jobs.append(entry(f'select:{task}', 'select', {'task': task},
                          deps=source_ids[task], resource='inline', priority=100))
        jobs.append(entry(f'prepare:{task}', 'prepare', {'task': task, 'output': str(Path(queue_root) / 'stress_cases' / f'{task}.json')},
                          resource='cpu', priority=100))
    for base in (RANK, CONTROLS):
        path = base / 'workflow.json'
        document = read(path)
        documents[str(path)] = document
        if document.get('phase') != 'waiting' or document.get('smoke_jobs'):
            raise ValueError('Initial migration currently expects recovery queues still waiting')
        for original in document['jobs']:
            key = f'{base.name}/{original["id"]}'
            location = {'manifest': str(path), 'field': 'jobs', 'key': original['id'], 'key_field': 'id'}
            if original['kind'] == 'test':
                deps = [f'{base.name}/{original["deps"][0]}']
                jobs.append(entry(key, 'recovery_test', original, location, deps, priority=35))
                continue
            task, name = original['task'], original['name']
            gate = [] if original['arm'] == 'A4' else [f'select:{task}']
            smoke_payload = copy.deepcopy(original)
            smoke_payload['output'] = str(ROOT / 'artifacts/smoke' / base.name / 'train' / name)
            smoke_payload['id'] = 'train:' + name
            smoke_key = f'{base.name}/smoke_train:{name}'
            smoke_location = {'manifest': str(path), 'field': 'smoke_jobs', 'key': 'train:'+name, 'key_field': 'id'}
            jobs.append(entry(smoke_key, 'smoke_train', smoke_payload, smoke_location,
                              gate + [f'prepare:{task}'], priority=30 if original['arm'] == 'A4' else 10))
            eval_payload = {'id': 'test:'+name, 'name': name, 'arm': original['arm'], 'task': task,
                            'kind': 'test', 'status': 'waiting', 'split': 'dev', 'limit': 6,
                            'output': str(ROOT / 'artifacts/smoke' / base.name / 'test' / name)}
            eval_key = f'{base.name}/smoke_test:{name}'
            eval_location = {'manifest': str(path), 'field': 'smoke_jobs', 'key': 'test:'+name, 'key_field': 'id'}
            jobs.append(entry(eval_key, 'smoke_test', eval_payload, eval_location, [smoke_key], priority=45))
            jobs.append(entry(key, 'recovery_train', original, location, [eval_key],
                              priority=25 if original['arm'] == 'A4' else 5))
    validate_graph(jobs)
    return {'schema_version': SCHEMA_VERSION, 'created_at': now(), 'status': 'ready',
            'host': 'suzy', 'allowed_gpus': list(range(1, 8)), 'jobs': jobs, 'documents': documents,
            'source_entries': source_entries, 'selected_sources': {}, 'reporting_errors': [],
            'policy': {'max_retries': 1, 'memory_headroom_gib': 1., 'cpu_workers': 1,
                       'A5_enabled': False, 'scientific_recipes_unchanged': True}}


def original_teacher(task):
    checkpoint = pin_best(SOURCES[task])
    if checkpoint is None:
        raise ValueError('Original teacher is incomplete')
    return checkpoint, read(Path(checkpoint) / 'metadata.json')['config']


def configure_recovery(state, job):
    """Materialize one task/arm as soon as its own source is pinned."""
    if job['payload'].get('config'):
        return
    payload = job['payload']
    task, arm = job['task'], payload['arm']
    if arm == 'A4':
        teacher, cfg = original_teacher(task)
        selected = {task: {'winner': {'name': 'original_' + task, 'teacher_checkpoint': teacher, 'config': cfg}}}
    else:
        selected = {task: state['selected_sources'][task]}
    root = RANK if job['id'].startswith(RANK.name + '/') else CONTROLS
    smoke = job['kind'] == 'smoke_train'
    output = ROOT / 'artifacts/smoke' / root.name if smoke else root
    native = next(j for j in materialize_jobs(output, selected, CACHE, smoke=smoke,
                                              arms=(arm,), tasks=(task,)) if j['kind'] == 'train')
    payload.update(native)
    payload['status'] = 'waiting'
    if smoke:
        # Eight large examples x two epochs -> four requested updates; stop at two.
        payload['config']['epochs'] = 2
        payload['config_sha256'] = config_fingerprint(payload['config'])
        payload['memory_cases'] = str(Path(state['queue_root']) / 'stress_cases' / f'{task}.json')
    payload['parent_checkpoint'] = str(Path(payload['init_adapter']).parent) if payload['init_adapter'] else None
    if payload['parent_checkpoint']:
        metadata = read(Path(payload['parent_checkpoint']) / 'metadata.json')
        payload['parent_step'] = metadata['global_step']
        payload['parent_epoch'] = metadata['global_step'] / max(1, read(Path(state['selected_sources'][task]['winner']['sft_run']) / 'report.json')['steps_per_epoch'])


def pin_evaluation(state, job):
    parent = {j['id']: j for j in state['jobs']}[job['deps'][0]]
    payload = job['payload']
    if job['kind'] == 'followup':
        if payload['kind'] == 'sft_test':
            run = parent['payload']['output']
        else:
            run = parent['payload']['run_dir']
    else:
        run = parent['payload']['run_dir']
    if (job['kind'] == 'smoke_test'
            and (payload.get('reload_latest') or parent['payload'].get('config', {}).get('lora_target') in ('decoder_encoder_ffn','decoder_encoder_linear'))):
        # A two-update smoke can have its best validation score at step zero.
        # Reload the updated adapter here; full official tests still pin best val.
        report = read(Path(run)/'report.json')
        checkpoint = str((Path(run)/'checkpoints/latest').resolve())
        metadata = read(Path(checkpoint)/'metadata.json')
        if (report.get('status') != 'paused' or report.get('global_step') != 2
                or report.get('test_set_used') is not False or metadata.get('global_step') != 2
                or metadata.get('config') != report.get('config')):
            raise ValueError('Full-model smoke reload must use its updated two-step checkpoint')
        payload['checkpoint_selection'] = 'latest_two_step_smoke_adapter'
    else:
        checkpoint = pin_best(run, allow_paused=job['kind'] == 'smoke_test')
    if checkpoint is None:
        raise ValueError('Finished parent has no validation-selected checkpoint')
    payload.update(checkpoint=checkpoint, checkpoint_identity=checkpoint_identity(checkpoint))
    payload.setdefault('split', 'dev' if job['kind'] == 'smoke_test' else 'test')
    payload.setdefault('limit', 6 if job['kind'] == 'smoke_test' else None)



def validate_export(payload):
    from smarthome_efficiency.pruning import sha256_file
    model = Path(payload['output'])
    report = read(model/'compression.json')
    expected = payload['request']
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Pruning export does not match its pinned request')
    if checkpoint_identity(expected['source_checkpoint']) != payload['source_identity']:
        raise ValueError('Pruning source checkpoint changed')
    if report.get('calibration', {}).get('sha256') != sha256_file(Path(expected['calibration_cache'])):
        raise ValueError('Pruning calibration cache changed')
    if payload.get('generate_calibration'):
        spec = report.get('calibration', {}).get('spec', {})
        if (spec.get('examples') != expected['calibration_samples'] or spec.get('split') != 'train'
                or spec.get('test_set_used') is not False
                or spec.get('source_checkpoint') != expected['source_checkpoint']
                or spec.get('source_metadata_sha256') != expected['source_metadata_sha256']
                or spec.get('source_adapter_sha256') != expected['source_adapter_sha256']
                or spec.get('calibration_seed') != expected['calibration_seed']):
            raise ValueError('Generated calibration provenance differs from the requested training-only sample')
    verification = report.get('verification', {})
    if (not all(verification.get(k) is True for k in
                ('passed', 'save_reload_class_logits_match', 'save_reload_generation_match'))
            or report.get('test_set_used') is not False or report.get('verification_split') != 'train'
            or (expected.get('budget_scope') not in ('total', 'separate') and report.get('encoder_tensors_unchanged') is not True)
            or report.get('recovery_updates') != 0):
        raise ValueError('Pruning export verification is incomplete')
    count = report['parameters_after']['language']
    cap = 2_000_000_000 if expected.get('budget_scope') == 'separate' else 1_999_999_999
    if count != payload['expected_language_parameters'] or not 0 < count <= cap:
        raise ValueError('Pruning export parameter count is outside the requested budget')
    config = read(model/'config.json')['text_config']
    if config['intermediate_size'] != expected['ffn_width'] or config['num_hidden_layers'] != 36:
        raise ValueError('Pruning export architecture differs from its request')
    if expected.get('budget_scope') in ('total', 'separate'):
        separate = expected.get('budget_scope') == 'separate'
        if separate:
            if expected['method'] == 'encoder_ffn_activation':
                from smarthome_efficiency.encoder_pruning import validate_separate_budget
                validate_separate_budget(report['parameters_before'], report['parameters_after'])
                if report.get('language_tensors_unchanged') is not True:
                    raise ValueError('Encoder-only export modified the LM')
            elif expected['method'] == 'multimodal_ffn_activation':
                from smarthome_efficiency.joint_pruning import validate_joint_budget
                validate_joint_budget(report['parameters_before'], report['parameters_after'])
                if report.get('joint_language_encoder_pruning') is not True:
                    raise ValueError('Missing direct joint-pruning provenance')
            else:
                raise ValueError('Unsupported separate-budget pruning method')
            if (report.get('parameter_limits') != {'language': 2_000_000_000, 'combined_encoders': 1_000_000_000}
                    or report.get('combined_encoder_parameters') != sum(report['parameters_after'][k] for k in ('audio_encoder','vision_encoder'))):
                raise ValueError('Separate-budget export lacks parameter limit evidence')
        hybrid = expected['method'] == 'multimodal_depth_ffn_activation'
        if (report['parameters_after'] != payload['expected_parameters']
                or (not separate and not 0 < report['parameters_after']['total'] < 2_000_000_000)
                or (not hybrid and report.get('non_ffn_tensors_unchanged') is not True)
                or (hybrid and report.get('retained_non_ffn_tensors_unchanged') is not True)
                or report.get('av_interfaces_unchanged') is not True
                or (not hybrid and report.get('all_layer_counts_unchanged') is not True)
                or report['calibration']['spec'].get('scope') != 'all_thinker_components'):
            raise ValueError('Full-model export exceeds its total budget or changes protected tensors/interfaces')
        full_config = read(model/'config.json')
        if (full_config['audio_config']['encoder_ffn_dim'] != expected['audio_ffn_width']
                or full_config['audio_config']['encoder_layers'] != expected.get('audio_layers',32)
                or full_config['vision_config']['intermediate_size'] != expected['vision_ffn_width']
                or full_config['vision_config']['depth'] != expected.get('vision_layers',32)):
            raise ValueError('Pruned encoder architecture differs from its request')
        if hybrid:
            a=report.get('kept_original_audio_layer_indices',[]);v=report.get('kept_original_vision_layer_indices',[])
            full=report.get('original_vision_full_attention_indices',[])
            mapped=[i for i,old in enumerate(v) if old in full]
            if (len(a)!=expected['audio_layers'] or len(v)!=expected['vision_layers']
                    or a!=sorted(set(a)) or v!=sorted(set(v)) or not {0,31}.issubset(a)
                    or not {0,7,15,23,31}.issubset(v) or full!=[7,15,23,31]
                    or full_config['vision_config']['fullatt_block_indexes']!=mapped):
                raise ValueError('Hybrid export lost protected encoder/global-attention layers')
    if not (model/'model.safetensors').exists() and not (model/'model.safetensors.index.json').exists():
        raise ValueError('Pruning export has no serialized weights')
    return {'report': str(model/'compression.json'), 'language_parameters': count,
            'total_parameters': report['parameters_after'].get('total'),
            'combined_encoder_parameters': (sum(report['parameters_after'][k] for k in ('audio_encoder','vision_encoder'))
                if all(k in report['parameters_after'] for k in ('audio_encoder','vision_encoder')) else None), 'model': str(model)}


def native_command(state, job, attempt):
    payload = job['payload']
    attempt = Path(attempt)
    if job['kind'] == 'benchmark_comparison':
        from smarthome_efficiency.pruning import sha256_file
        by_id={j['id']:j for j in state['jobs']}
        cases=[]
        for task,key in payload['test_jobs'].items():
            evaluation=by_id[key]
            if evaluation['status']!='completed':raise ValueError('Comparison requires completed official evaluations')
            training=by_id[evaluation['deps'][0]]
            cases.append({'task':task,'test_report':str(Path(evaluation['payload']['run_dir'])/'report.json'),
                          'training_report':str(Path(training['payload']['run_dir'])/'report.json')})
        cases.append(payload['mer_reference'])
        request={k:payload[k] for k in ('baseline_catalog','baseline_catalog_sha256','reference_training_report','reference_training_report_sha256')}
        request['cases']=cases
        path=attempt/'comparison_request.json'
        from smarthome_efficiency.queueing import atomic_json
        atomic_json(path,request);payload['request_sha256']=sha256_file(path)
        return [str(PROJECT_ENV/'bin/python'),'-B',str(ROOT/'scripts/compare_project_benchmarks.py'),
                '--request',str(path),'--output',payload['output']],{'num_threads':2}
    if job['kind'] == 'merge_export':
        parent = {j['id']:j for j in state['jobs']}.get(payload.get('training_job'))
        if parent is not None and parent['status'] != 'completed':
            raise ValueError('Merge parent training is incomplete')
        source_run = parent['payload']['run_dir'] if parent else payload['source_run']
        source = pin_best(source_run)
        if source is None: raise ValueError('Merge parent has no validation winner')
        identity = checkpoint_identity(source)
        if payload.get('source_identity') not in (None, identity): raise ValueError('Pinned merge source changed')
        cfg = read(Path(source)/'metadata.json')['config']
        from smarthome_efficiency.pruning import sha256_file
        digest = sha256_file(Path(cfg['model'])/'compression.json')
        if payload.get('source_base_manifest_sha256') not in (None,digest): raise ValueError('Pinned merge base changed')
        payload.update(source_checkpoint=source,source_identity=identity,source_base_manifest_sha256=digest)
        return [str(PROJECT_ENV/'bin/python'),'-B','-u',str(ROOT/'scripts/export_recovered_model.py'),
                '--source-checkpoint',source,'--output',payload['output']],cfg
    if job['kind'] == 'checkpoint_average':
        parent = {j['id']:j for j in state['jobs']}.get(payload.get('training_job'))
        source = parent['payload']['run_dir'] if parent else payload['source_run']
        selected = pin_best(source)
        identity = checkpoint_identity(selected)
        if payload.get('source_identity') not in (None, identity):
            raise ValueError('Pinned averaging parent changed')
        payload.update(source_run=source,source_identity=identity,run_dir=payload['output'])
        cfg = read(Path(source)/'report.json')['config']
        return [str(PROJECT_ENV/'bin/python'),'-B','-u',str(ROOT/'scripts/select_averaged_checkpoint.py'),
                '--source-run',source,'--output',payload['output']],cfg
    if payload.get('pin_generated_caches'):
        from smarthome_efficiency.cache_extension import pin_generated_caches
        pin_generated_caches(payload)
    if payload.get('expected_teacher_cache_sha256'):
        from smarthome_efficiency.pruning import sha256_file
        if sha256_file(Path(payload['config']['teacher_cache'])) != payload['expected_teacher_cache_sha256']:
            raise ValueError('Pinned class teacher cache changed before launch')
    if payload.get('expected_feature_cache_sha256'):
        from smarthome_efficiency.pruning import sha256_file
        path = Path(payload['config']['feature_teacher_cache'])/'cache.json'
        if sha256_file(path) != payload['expected_feature_cache_sha256']:
            raise ValueError('Pinned feature cache changed before launch')
    if job['kind'] == 'prune_export':
        from smarthome_efficiency.pruning import sha256_file
        request = payload['request']
        if checkpoint_identity(request['source_checkpoint']) != payload['source_identity']:
            raise ValueError('Pinned pruning source changed')
        if payload.get('calibration_sha256') is not None:
            if sha256_file(Path(request['calibration_cache'])) != payload['calibration_sha256']:
                raise ValueError('Pinned pruning calibration changed')
        elif not payload.get('generate_calibration'):
            raise ValueError('A pruning job must pin or explicitly generate its calibration cache')
        command = [str(PROJECT_ENV/'bin/python'), '-B', '-u', str(ROOT/'scripts/prune_model.py'),
                   '--source-checkpoint', request['source_checkpoint'], '--method', request['method'],
                   '--ffn-width', str(request['ffn_width']), '--output', payload['output'],
                   '--calibration-cache', request['calibration_cache'],
                   '--calibration-samples', str(request['calibration_samples']),
                   '--calibration-seed', str(request['calibration_seed']), '--reuse']
        if request.get('budget_scope') in ('total', 'separate'):
            command += ['--audio-ffn-width', str(request['audio_ffn_width']),
                        '--vision-ffn-width', str(request['vision_ffn_width']), '--budget-scope', request['budget_scope']]
            source_config = read(Path(request['source_checkpoint'])/'metadata.json')['config']
            base = Path(source_config['model'])
            if request.get('budget_scope') == 'separate' and request['method'] == 'multimodal_ffn_activation':
                from smarthome_efficiency.pruning import processor_hashes
                if (str(base.resolve()) != payload['source_base_snapshot'] or (base/'compression.json').exists()
                        or sha256_file(base/'config.json') != payload['source_base_config_sha256']
                        or processor_hashes(base) != payload['source_processor_files_sha256']):
                    raise ValueError('Joint pruning must use the pinned original unpruned snapshot')
            elif sha256_file(base/'compression.json') != payload['source_base_manifest_sha256']:
                raise ValueError('Pinned compact source base changed')
        if request['method'] == 'multimodal_depth_ffn_activation':
            command += ['--audio-layers',str(request['audio_layers']),'--vision-layers',str(request['vision_layers'])]
        return command, read(Path(request['source_checkpoint'])/'metadata.json')['config']
    if job['kind'] == 'encoder_cache':
        if checkpoint_identity(payload['teacher_checkpoint']) != payload['source_identity']:
            raise ValueError('Pinned encoder teacher checkpoint changed')
        cfg = read(Path(payload['teacher_checkpoint'])/'metadata.json')['config']
        command = [str(PROJECT_ENV/'bin/python'), '-B', '-u', str(ROOT/'scripts/cache_encoder_features.py'),
                   '--teacher-checkpoint', payload['teacher_checkpoint'], '--output', payload['output'], '--resume']
        if payload.get('mer_partition'):
            cfg = dict(cfg, mer_partition=payload['mer_partition'])
            command += ['--mer-partition', payload['mer_partition']]
        return command, cfg
    if job['kind'] == 'prepare':
        if payload.get('config'):
            cfg = payload['config']
            if cfg['task'] != job['task']:
                raise ValueError('Input preparation task/config mismatch')
        else:
            teacher, cfg = original_teacher(job['task'])
        path = attempt / 'config.yaml'
        import yaml
        path.write_text(yaml.safe_dump(DEFAULTS | cfg, sort_keys=False))
        command = [str(PROJECT_ENV/'bin/python'), '-B', '-u', str(ROOT/'scripts/prepare_memory_cases.py'),
                   '--config', str(path), '--output', payload['output']]
        if payload.get('require_teacher_disagreement'):
            command.append('--require-teacher-disagreement')
        return command, cfg
    if job['kind'] == 'followup' and payload['kind'] == 'teacher':
        if payload.get('source_identity') and checkpoint_identity(payload['teacher_checkpoint']) != payload['source_identity']:
            raise ValueError('Pinned logit teacher checkpoint changed')
        cfg = read(Path(payload['teacher_checkpoint'])/'metadata.json')['config']
        command = [str(PROJECT_ENV/'bin/python'), '-B', '-u', str(ROOT/'scripts/cache_teacher_logits.py'),
                   '--teacher-checkpoint', payload['teacher_checkpoint'], '--output', payload['output'], '--resume']
        if payload.get('mer_partition'):
            cfg = dict(cfg, mer_partition=payload['mer_partition'])
            command += ['--mer-partition', payload['mer_partition']]
        return command, cfg
    is_eval = job['kind'] in ('recovery_test', 'smoke_test') or (job['kind'] == 'followup' and payload['kind'].endswith('test'))
    if is_eval:
        pin_evaluation(state, job)
        if checkpoint_identity(payload['checkpoint']) != payload['checkpoint_identity']:
            raise ValueError('Evaluation checkpoint changed')
        run = Path(payload['output']) / f'attempt-{len(job["attempts"]):03d}'
        payload['run_dir'] = str(run)
        command = [str(PROJECT_ENV/'bin/python'), '-B', '-u', str(ROOT/'scripts/evaluate.py'),
                   '--checkpoint', payload['checkpoint'], '--split', payload['split'], '--modes', 'both', '--output', str(run)]
        if payload.get('limit') is not None:
            command += ['--limit', str(payload['limit'])]
        return command, read(Path(payload['checkpoint'])/'metadata.json')['config']
    if job['kind'] in ('recovery_train', 'smoke_train'):
        configure_recovery(state, job)
    cfg = payload['config']
    if payload.get('init_adapter') and checkpoint_identity(Path(payload['init_adapter']).parent) != payload['init_checkpoint_identity']:
        raise ValueError('Pinned parent adapter changed')
    resume = None
    if payload.get('run_dir') and (Path(payload['run_dir'])/'checkpoints/latest').exists():
        resume = (Path(payload['run_dir'])/'checkpoints/latest').resolve()
        from smarthome_efficiency.training import validate_resume
        saved = read(resume/'metadata.json')
        validate_resume(saved['config'], cfg, saved['world_size'], 1)
        run = Path(payload['run_dir'])
    else:
        run = Path(payload['output']) / f'attempt-{len(job["attempts"]):03d}'
    payload['run_dir'] = str(run)
    path = attempt/'config.yaml'
    import yaml
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = build_command({'kind': 'train', 'init_adapter': payload.get('init_adapter')},
                            run, path, job['kind'] == 'smoke_train', resume)
    if job['kind'] == 'smoke_train':
        command += ['--train-ids', payload['memory_cases']]
    return command, cfg


def validate_native(job):
    payload = job['payload']
    if job['kind'] == 'benchmark_comparison':
        report=read(payload['output'])
        if (report.get('status')!='completed' or report.get('request_sha256')!=payload['request_sha256']
                or report.get('test_set_used') is not True or report.get('checkpoint_selection_uses_test') is not False
                or {r['task'] for r in report.get('results',[])}!={'mintrec2','mintrec20','mer6'}):
            raise ValueError('Incomplete shared-recipe benchmark comparison')
        return {'report':payload['output'],'checks':report['checks'],'main_project_performance_targets_met':report['main_project_performance_targets_met'],'requested_verification_targets_met':report['requested_verification_targets_met']}
    if job['kind'] == 'merge_export':
        from export_recovered_model import validate_merge_export
        return validate_merge_export(payload['output'],payload['source_checkpoint'],payload['expected_parameters'])
    if job['kind'] == 'checkpoint_average':
        report=read(Path(payload['run_dir'])/'report.json')
        if (report['status']!='completed' or report['test_set_used'] is not False
                or report['source_run']!=payload['source_run'] or not report.get('dense_architecture_unchanged')
                or report['dev_count']!=payload['dev_count']):
            raise ValueError('Invalid averaged-checkpoint validation selection')
        checkpoint=pin_best(payload['run_dir'])
        if checkpoint_identity(checkpoint)!=report['checkpoint_identity']:
            raise ValueError('Selected averaged checkpoint changed')
        return {'report':str(Path(payload['run_dir'])/'report.json'),'best_score':report['best_score'],
                'selected_candidate':report['selected_candidate'],'checkpoint':checkpoint}
    if job['kind'] == 'encoder_cache':
        from smarthome_efficiency.pruning import sha256_file
        folder=Path(payload['output']);cached=read(folder/'cache.json');request=read(folder/'request.json')
        if (cached.get('status')!='completed' or cached.get('num_examples')!=payload['expected_examples']
                or len(cached['rows'])!=payload['expected_examples'] or cached['spec']!=request['spec']
                or cached['spec']['teacher']['split']!='train' or cached['spec']['teacher']['test_set_used'] is not False
                or cached['spec']['teacher']['teacher_checkpoint']!=payload['teacher_checkpoint']
                or checkpoint_identity(payload['teacher_checkpoint'])!=payload['source_identity']
                or set(cached['rows'])!=set(request['record_fingerprints'])
                or any(row['sample_sha256']!=request['record_fingerprints'][key]
                       or not (folder/'features'/row['file']).is_file() for key,row in cached['rows'].items())):
            raise ValueError('Incomplete or mismatched frozen encoder feature cache')
        return {'cache':str(folder/'cache.json'),'cache_sha256':sha256_file(folder/'cache.json'),
                'examples':cached['num_examples'],'feature_bytes':cached['feature_bytes']}
    if job['kind'] == 'prune_export':
        return validate_export(payload)
    if job['kind'] == 'sft':
        from launch_suzy_sweep import completion
        result = completion(payload, 0, False)
        if result['status'] != 'completed':
            raise ValueError('Imported SFT report did not verify completion')
        return result
    if job['kind'] == 'followup':
        return completed_result(payload, False)
    if job['kind'] == 'prepare':
        result = read(payload['output'])
        if result['status'] != 'completed' or result['split'] != 'train' or result['test_set_used'] is not False or len(result['ids']) != 8:
            raise ValueError('Invalid memory probe manifest')
        if payload.get('require_teacher_disagreement'):
            coverage = result.get('teacher_coverage', {})
            wrong, correct = coverage.get('disagreement_ids', []), coverage.get('agreement_ids', [])
            from smarthome_efficiency.pruning import sha256_file
            if (not wrong or not correct or set(wrong) & set(correct)
                    or sorted(wrong + correct) != sorted(result['ids'])
                    or coverage.get('teacher_cache_sha256') != sha256_file(Path(payload['config']['teacher_cache']))):
                raise ValueError('Memory preflight lacks verified teacher-agreement/disagreement coverage')
        return {'manifest': payload['output'], 'examples_indexed': result['examples_indexed']}
    result = validate_result(payload, job['kind'].startswith('smoke_'))
    if payload.get('expected_feature_cache_sha256'):
        report = read(Path(payload['run_dir'])/'report.json')
        if report.get('feature_cache_sha256') != payload['expected_feature_cache_sha256']:
            raise ValueError('Run did not use its pinned encoder-feature cache')
    if job['kind'] == 'smoke_train' and payload['config'].get('lora_target') in ('decoder_encoder_ffn','decoder_encoder_linear'):
        run = Path(payload['run_dir'])
        events = [__import__('json').loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        coverage = [e['components'] for e in events if e['event'] == 'adapter_gradient_coverage']
        if (len(coverage) != 1 or set(coverage[0]) != {'language','audio_encoder','vision_encoder'}
                or any(v['gradient_l2'] <= 0 or v['tensors_with_gradient'] < 1 for v in coverage[0].values())):
            raise ValueError('Smoke did not verify gradients through all three components')
        from safetensors import safe_open
        import torch
        with safe_open(Path(read(run/'report.json')['latest_checkpoint'])/'adapter/adapter_model.safetensors',
                       framework='pt', device='cpu') as weights:
            for prefix in ('model.layers.', 'audio_tower.layers.', 'visual.blocks.'):
                keys = [k for k in weights.keys() if prefix in k and 'lora_B' in k]
                if not keys or not any(torch.count_nonzero(weights.get_tensor(k)).item() for k in keys):
                    raise ValueError('Serialized smoke adapter did not update ' + prefix)
        result['adapter_gradient_coverage'] = coverage[0]
        if payload['config'].get('feature_teacher_cache'):
            updates=[e for e in events if e['event']=='train']
            import math
            from native_encoder_preflight import validate_feature_smoke
            validate_feature_smoke(payload, updates)
            report=read(run/'report.json')
            if not report.get('feature_cache_sha256'):
                raise ValueError('Feature-KD checkpoint provenance does not pin its cache')
            result['feature_cache_sha256']=report['feature_cache_sha256']
        if payload['config'].get('distillation_disagreement_alpha') is not None:
            from smarthome_efficiency.disagreement_preflight import validate_disagreement_smoke
            probe = read(payload['memory_cases'])
            training_manifest = read(run/'data_manifest.json')
            report = read(run/'report.json')
            result['teacher_disagreement_coverage'] = validate_disagreement_smoke(
                payload['config'], events, probe, training_manifest['train'], report['teacher_cache_sha256'])
    return result

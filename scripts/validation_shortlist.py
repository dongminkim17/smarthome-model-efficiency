"""Validation-only official-test selection, by single run or repeated-seed group."""
from pathlib import Path
import math
from smarthome_efficiency.queueing import TERMINAL, atomic_json, now, read_json, fingerprint
from smarthome_efficiency.pruning import sha256_file


def _group_config(config):
    """Ignore seed/execution grouping, while retaining the actual effective batch."""
    value = dict(config)
    value.pop('seed', None)
    value['effective_batch_per_rank'] = value.pop('micro_batch_size') * value.pop('gradient_accumulation_steps')
    value.pop('gradient_checkpointing', None)
    return value


def _declared_groups(job):
    """Validate immutable group membership before inspecting any scores."""
    groups = job['payload'].get('groups')
    if groups is None:
        return None
    seeds = job['payload'].get('required_seeds')
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds) or not isinstance(groups, list) or not groups):
        raise ValueError('Grouped shortlist requires distinct integer seeds and nonempty groups')
    names, members = set(), []
    for group in groups:
        name = group.get('name')
        if not isinstance(name, str) or not name or name in names or not isinstance(group.get('config'), dict):
            raise ValueError('Grouped shortlist needs unique names and explicit configuration contracts')
        names.add(name)
        rows = group.get('members')
        if (not isinstance(rows, list) or len(rows) != len(seeds)
                or any(not isinstance(r, dict) or not isinstance(r.get('job'), str)
                       or type(r.get('seed')) is not int for r in rows)
                or sorted(r['seed'] for r in rows) != sorted(seeds)):
            raise ValueError('Every group must declare exactly one run per required seed')
        members.extend(r['job'] for r in rows)
        _group_config(group['config'])
    if len(members) != len(set(members)) or sorted(members) != sorted(job['deps']):
        raise ValueError('Grouped shortlist dependencies must exactly match unique declared members')
    for field in ('expected_data_manifest_sha256', 'expected_teacher_cache_sha256', 'expected_feature_cache_sha256'):
        if not isinstance(job['payload'].get(field), str) or not job['payload'][field]:
            raise ValueError('Grouped shortlist requires pinned data and teacher-cache identities')
    if type(job['payload'].get('train_count')) is not int or job['payload']['train_count'] < 1:
        raise ValueError('Grouped shortlist requires the complete training count')
    return groups


def select_validation_shortlist(state, job):
    """Wait for terminal candidates; select only from full validation reports.

    Optional explicit groups rank the mean best-dev score across every required
    seed. An incomplete group is excluded, never averaged over surviving seeds.
    Historical single-run selection keeps its original behavior.
    """
    from launch_pruning_followup import checkpoint_identity, pin_best
    by_id = {item['id']: item for item in state['jobs']}
    top_k = job['payload']['top_k']
    if type(top_k) is not int or top_k < 1:
        raise ValueError('Shortlist size must be a positive integer')
    groups = _declared_groups(job)
    declarations = ({m['job']: (g, m) for g in groups for m in g['members']} if groups else {})
    candidates, skipped = [], []
    for key in job['deps']:
        candidate = by_id[key]
        if candidate['kind'] not in ('recovery_train', 'checkpoint_average') or candidate['task'] != job['task']:
            raise ValueError('Shortlist candidates must be same-task scientific training jobs')
        if candidate['status'] not in TERMINAL:
            raise ValueError('A candidate is still active or waiting')
        if candidate['status'] != 'completed':
            skipped.append({'job': key, 'status': candidate['status']})
            continue
        payload = candidate['payload']
        run = Path(payload['run_dir'])
        report = read_json(run/'report.json')
        cfg = report['config']
        score = report.get('best_score')
        if (report.get('status') != 'completed' or report.get('test_set_used') is not False
                or report.get('dev_count') != job['payload']['dev_count']
                or cfg.get('dev_limit') is not None or cfg.get('train_limit') is not None
                or cfg.get('selection_mode') != 'logit' or cfg.get('task') != job['task']
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Shortlist requires complete validation-only training reports')
        if groups:
            group, member = declarations[key]
            if (candidate['kind'] != 'recovery_train' or cfg.get('seed') != member['seed']
                    or report.get('train_count') != job['payload']['train_count']
                    or report.get('world_size') != 1
                    or fingerprint(_group_config(cfg)) != fingerprint(_group_config(group['config']))
                    or sha256_file(run/'data_manifest.json') != job['payload']['expected_data_manifest_sha256']
                    or report.get('teacher_cache_sha256') != job['payload']['expected_teacher_cache_sha256']
                    or report.get('feature_cache_sha256') != job['payload']['expected_feature_cache_sha256']):
                raise ValueError('Grouped candidate changed its declared seed, recipe, data, or teacher targets')
        checkpoint = pin_best(payload['run_dir'])
        if checkpoint is None:
            raise ValueError('Completed candidate lacks its validation-selected checkpoint')
        row = {'job': key, 'name': payload['name'], 'validation_score': score,
               'checkpoint': checkpoint, 'checkpoint_identity': checkpoint_identity(checkpoint)}
        if groups:
            row.update(group=group['name'], seed=member['seed'])
        candidates.append(row)
    if not candidates and not groups:
        raise ValueError('No successful training candidates available for official testing')
    candidates.sort(key=lambda row: (-row['validation_score'], row['job']))
    decision = {'schema_version': 1, 'selected_at': now(), 'metric': 'validation_logit_weighted_f1',
                'task': job['task'], 'requested_top_k': top_k,
                'ranking': candidates, 'skipped_failed_candidates': skipped,
                'test_results_used': False, 'tie_break': 'ascending immutable training job ID'}
    if groups:
        lookup = {r['job']: r for r in candidates}
        ranked, incomplete = [], []
        for group in groups:
            missing = [m['job'] for m in group['members'] if m['job'] not in lookup]
            if missing:
                incomplete.append({'group': group['name'], 'missing_successful_members': missing})
                continue
            rows = [lookup[m['job']] for m in sorted(group['members'], key=lambda m: m['seed'])]
            ranked.append({'group': group['name'], 'mean_validation_score': math.fsum(r['validation_score'] for r in rows)/len(rows),
                           'members': rows})
        ranked.sort(key=lambda row: (-row['mean_validation_score'], row['group']))
        winners = ranked[:top_k]
        selected = [r for group in winners for r in group['members']]
        decision.update(selection_unit='complete_repeated_seed_group', required_seeds=job['payload']['required_seeds'],
                        selected_groups=[r['group'] for r in winners], group_ranking=ranked,
                        excluded_incomplete_groups=incomplete,
                        tie_break='ascending immutable group name; members by seed')
    else:
        selected = candidates[:top_k]
    decision['selected'] = [row['job'] for row in selected]
    atomic_json(job['payload']['output'], decision)
    selected_ids = set(decision['selected'])
    for evaluation in state['jobs']:
        if evaluation['kind'] != 'recovery_test' or evaluation['payload'].get('shortlist_job') != job['id']:
            continue
        parent = evaluation['deps'][0]
        evaluation['payload']['shortlist_selected'] = parent in selected_ids
        if parent not in selected_ids and evaluation['status'] in ('waiting', 'ready'):
            reason = 'Not selected by the validation-only official-test shortlist'
            evaluation.update(status='cancelled', reason=reason, finished_at=now())
            evaluation['payload'].update(status='cancelled', cancellation_reason=reason)
    state.setdefault('shortlists', {})[job['id']] = decision
    return decision

"""Verify a bounded GPU smoke actually exercises both teacher-guidance branches."""
import math


def validate_disagreement_smoke(config, events, probe, train_records, cache_sha256):
    if probe.get('split') != 'train' or probe.get('test_set_used') is not False:
        raise ValueError('Teacher-gating probe must use training data only')
    coverage = probe.get('teacher_coverage', {})
    wrong, correct = coverage.get('disagreement_ids', []), coverage.get('agreement_ids', [])
    ids = probe.get('ids', [])
    if (len(ids) != 8 or len(set(ids)) != 8 or not wrong or not correct
            or set(wrong) & set(correct) or sorted(wrong+correct) != sorted(ids)
            or sorted(record['id'] for record in train_records) != sorted(ids)
            or coverage.get('teacher_cache_sha256') != cache_sha256):
        raise ValueError('Smoke teacher-coverage provenance does not match its eight training clips')
    updates = [e for e in events if e.get('event') == 'train']
    if len(updates) != 2 or [e.get('global_step') for e in updates] != [1, 2]:
        raise ValueError('Teacher-gating preflight requires exactly two saved updates')
    nominal, gated = config['distillation_alpha'], config['distillation_disagreement_alpha']
    total, errors, weighted_alpha = 0, 0, 0.
    for e in updates:
        n, count = e.get('examples'), e.get('teacher_disagreement_count')
        rate, alpha = e.get('teacher_disagreement_rate'), e.get('distillation_alpha_effective_mean')
        if (type(n) is not int or n != 4 or type(count) is not int or not 0 <= count <= n
                or not isinstance(rate, (float,int)) or not math.isfinite(rate)
                or not isinstance(alpha, (float,int)) or not math.isfinite(alpha)
                or abs(rate-count/n) > 1e-7
                or abs(alpha-((n-count)*nominal+count*gated)/n) > 1e-7):
            raise ValueError('GPU smoke did not apply the declared per-example distillation weights')
        total += n; errors += count; weighted_alpha += n*alpha
    if total != 8 or errors != len(wrong):
        raise ValueError('GPU smoke did not exercise the expected teacher-disagreement training clips')
    return {'examples': total, 'teacher_disagreements': errors,
            'mean_effective_alpha': weighted_alpha/total,
            'teacher_cache_sha256': cache_sha256, 'both_branches_exercised': True}

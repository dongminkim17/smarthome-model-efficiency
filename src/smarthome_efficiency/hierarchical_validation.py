"""Fixed MIntRec fine-to-coarse projection; consumes saved validation rows only."""
from __future__ import annotations
import math
from .data import LABELS, MINTREC_COARSE
from .evaluation import classification_metrics


def project_validation_rows(rows, expected_records, labels):
    if list(labels) != list(LABELS['mintrec20']):
        raise ValueError('Fine-class logit order must match the saved MIntRec taxonomy')
    expected = {r['id']: r['label'] for r in expected_records}
    if len(expected) != len(expected_records) or not expected:
        raise ValueError('Validation manifest contains missing or duplicate IDs')
    if len(rows) != len(expected) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Prediction rows must cover the full validation split exactly once')
    projected = []
    for row in rows:
        if row['id'] not in expected or expected[row['id']] != row['true_label']:
            raise ValueError('Prediction ID/label differs from the saved validation manifest')
        scores = row.get('logit_scores', [])
        if (len(scores) != len(labels) or any(isinstance(v, bool) or not isinstance(v, (float,int))
                                             or not math.isfinite(v) for v in scores)):
            raise ValueError('Expected twenty finite saved fine-class logits')
        fine = labels[max(range(len(scores)), key=scores.__getitem__)]
        if row.get('logit_predicted_label') != fine:
            raise ValueError('Saved fine prediction differs from logit order/argmax')
        weights = [math.exp(v-max(scores)) for v in scores]
        denominator = sum(weights)
        probability = {c: sum(w for w,label in zip(weights,labels) if MINTREC_COARSE[label]==c)/denominator
                       for c in LABELS['mintrec2']}
        coarse = max(LABELS['mintrec2'], key=probability.__getitem__)
        projected.append({'id': row['id'], 'true_label': MINTREC_COARSE[row['true_label']],
                          'probability_sum_prediction': coarse, 'coarse_probabilities': probability,
                          'hard_map_prediction': MINTREC_COARSE[fine],
                          'greedy_map_prediction': MINTREC_COARSE.get(row.get('greedy_predicted_label'))})
    truth = [r['true_label'] for r in projected]
    metrics = {method: classification_metrics(truth, [r[key] for r in projected], LABELS['mintrec2'])
               for method,key in [('probability_sum','probability_sum_prediction'),
                                  ('hard_map','hard_map_prediction'), ('greedy_map','greedy_map_prediction')]}
    return projected, metrics

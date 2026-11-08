"""Explicit prediction protocols and classification metrics, including invalid output.

The greedy parser accepts exactly one option letter (case insensitive), optional
parentheses, an optional ``Answer`` or ``Option`` prefix with optional colon, and
one terminal full stop or exclamation mark. Surrounding/repeated whitespace is
normalized. It does not extract letters from explanations, accept class names,
resolve multiple options, or fall back to logit predictions.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Sequence

PARSER_PROTOCOL = (
    'One option letter, case insensitive; optional Answer/Option prefix and colon, '
    'parentheses, and one terminal period/exclamation mark. No explanation, class-name, '
    'substring, or logit fallback.'
)
INVALID_LABEL = '<invalid>'


def normalize_modes(modes: str | Sequence[str]) -> tuple[str, ...]:
    """Return stable mode order, rejecting unknown or empty choices."""
    values = [modes] if isinstance(modes, str) else list(modes)
    expanded = set()
    for value in values:
        if value == 'both':
            expanded.update(('logit', 'greedy'))
        elif value in ('logit', 'greedy'):
            expanded.add(value)
        else:
            raise ValueError(f'Unknown evaluation mode: {value!r}')
    if not expanded:
        raise ValueError('At least one evaluation mode is required')
    return tuple(value for value in ('logit', 'greedy') if value in expanded)


def _check_labels(labels: Sequence[str]) -> list[str]:
    result = list(labels)
    if not 2 <= len(result) <= 26 or len(set(result)) != len(result):
        raise ValueError('Expected 2 to 26 distinct task labels')
    if INVALID_LABEL in result:
        raise ValueError(f'{INVALID_LABEL!r} is reserved for invalid predictions')
    return result


def parse_generated_label(text: str, labels: Sequence[str]) -> dict[str, str | None]:
    """Parse only the documented output grammar; malformed outputs remain invalid."""
    labels = _check_labels(labels)
    if not isinstance(text, str):
        raise TypeError('Generated output must be a string')
    normalized = ' '.join(text.split()).strip()
    if not normalized:
        return {'predicted_label': None, 'invalid_reason': 'empty_output'}
    match = re.fullmatch(
        r'(?:(?:answer|option)(?:\s*:\s*|\s+))?(?:\(([A-Za-z])\)|([A-Za-z]))[.!]?',
        normalized, flags=re.IGNORECASE,
    )
    if match is None:
        option_letters = re.findall(r'\b([A-Za-z])\b', normalized)
        unique_valid = {letter.upper() for letter in option_letters if 0 <= ord(letter.upper()) - 65 < len(labels)}
        reason = 'ambiguous_options' if len(unique_valid) > 1 else 'unsupported_format'
        return {'predicted_label': None, 'invalid_reason': reason}
    letter = (match.group(1) or match.group(2)).upper()
    index = ord(letter) - 65
    if not 0 <= index < len(labels):
        return {'predicted_label': None, 'invalid_reason': 'option_out_of_range'}
    return {'predicted_label': labels[index], 'invalid_reason': None}


def classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str | None],
    labels: Sequence[str],
) -> dict:
    """Compute metrics with every invalid prediction retained as an error.

    Confusion rows follow ``labels`` and columns follow ``labels + ['<invalid>']``.
    Macro F1 includes every configured class, with zero division defined as zero.
    """
    labels = _check_labels(labels)
    truth, predictions = list(true_labels), list(predicted_labels)
    if len(truth) != len(predictions):
        raise ValueError('Truth and prediction lengths differ')
    label_to_index = {label: index for index, label in enumerate(labels)}
    if any(label not in label_to_index for label in truth):
        raise ValueError('Ground truth contains an unknown label')
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for target, prediction in zip(truth, predictions):
        column = label_to_index.get(prediction, len(labels))
        matrix[label_to_index[target]][column] += 1
    n = len(truth)
    invalid = sum(row[-1] for row in matrix)
    per_class = {}
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'support': support, 'predicted_count': predicted_count,
        }
    predicted_counts = Counter(prediction if prediction in label_to_index else INVALID_LABEL for prediction in predictions)
    return {
        'n': n,
        'accuracy': sum(matrix[index][index] for index in range(len(labels))) / n if n else 0.0,
        'weighted_f1': sum(row['f1'] * row['support'] for row in per_class.values()) / n if n else 0.0,
        'macro_f1': sum(row['f1'] for row in per_class.values()) / len(labels),
        'invalid_count': invalid,
        'invalid_rate': invalid / n if n else 0.0,
        'predicted_counts': dict(predicted_counts),
        'per_class': per_class,
        'confusion_matrix': matrix,
        'confusion_rows': labels,
        'confusion_columns': labels + [INVALID_LABEL],
    }


def prediction_agreement(
    logit_labels: Sequence[str], greedy_labels: Sequence[str | None],
) -> dict:
    """Agreement keeps invalid greedy answers in the overall denominator."""
    logit, greedy = list(logit_labels), list(greedy_labels)
    if len(logit) != len(greedy):
        raise ValueError('Logit and greedy prediction lengths differ')
    if any(label is None or label == INVALID_LABEL for label in logit):
        raise ValueError('Logit predictions must be valid labels')
    valid = [b is not None and b != INVALID_LABEL for b in greedy]
    count = sum(is_valid and a == b for a, b, is_valid in zip(logit, greedy, valid))
    valid_count = sum(valid)
    return {
        'n': len(logit), 'agreement_count': count,
        'agreement': count / len(logit) if logit else 0.0,
        'greedy_invalid_count': len(logit) - valid_count,
        'agreement_on_valid': count / valid_count if valid_count else 0.0,
    }

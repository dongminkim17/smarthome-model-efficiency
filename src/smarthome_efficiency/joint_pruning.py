"""Separate LM/encoder caps for direct, joint pruning of an unpruned teacher.

Surgery and activation calibration use the existing all-component FFN pipeline.
Encoder-only preservation checks remain separate and unchanged.
"""
from .encoder_pruning import LANGUAGE_LIMIT, ENCODER_LIMIT, encoder_parameters
from .multimodal_pruning import expected_counts


def validate_joint_budget(before, after):
    if (before != expected_counts(11008, 5120, 3420)
            or not 0 < after['language'] <= LANGUAGE_LIMIT
            or not 0 < encoder_parameters(after) <= ENCODER_LIMIT
            or after['language'] >= before['language']
            or encoder_parameters(after) >= encoder_parameters(before)
            or after['other'] != before['other']):
        raise ValueError('Joint export must directly reduce the original unpruned teacher LM and encoders to separate <=2B/<=1B caps')

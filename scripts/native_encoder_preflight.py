"""Feature supervision checks for compact and unchanged original encoders."""
import math
from pathlib import Path
from smarthome_efficiency.queueing import read_json


def validate_feature_smoke(payload, updates):
    native = payload.get('original_encoders') is True
    if native:
        report=read_json(Path(payload['config']['model'])/'compression.json')
        counts=report['parameters_after']
        if (report.get('method')!='merged_lm_recovery' or report.get('encoder_tensors_unchanged') is not True
                or counts['audio_encoder']!=637676544 or counts['vision_encoder']!=668684288):
            raise ValueError('Zero feature distance requires verified original encoders')
    if not updates:raise ValueError('Feature supervision has no training updates')
    for event in updates:
        for key in ('audio_feature_loss','vision_feature_loss'):
            value=event.get(key,float('nan'))
            # Matching unchanged encoders initially have zero cosine distance;
            # float32 cosine reduction can also yield a tiny negative roundoff.
            if not math.isfinite(value) or (value < -1e-6 if native else value <= 0):
                raise ValueError('Invalid feature-KD smoke supervision')

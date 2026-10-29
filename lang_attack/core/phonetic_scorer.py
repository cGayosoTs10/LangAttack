"""Phonetic scorer facade for Wav2Vec 2.0 / GOP alignment."""

from lang_attack.core.phonetics import (
    ARTICULATORY_GUIDANCE,
    ARTICULATORY_GUIDANCE_CATALOG,
    SYMBOL_TO_ARTICULATORY,
    PhoneticEvaluator,
    Wav2VecEvaluator,
    decode_audio_samples,
    evaluate_speech,
    extract_acoustic_features,
)

__all__ = [
    "ARTICULATORY_GUIDANCE",
    "ARTICULATORY_GUIDANCE_CATALOG",
    "PhoneticEvaluator",
    "SYMBOL_TO_ARTICULATORY",
    "Wav2VecEvaluator",
    "decode_audio_samples",
    "evaluate_speech",
    "extract_acoustic_features",
]

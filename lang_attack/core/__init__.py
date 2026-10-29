"""Core AI inference, speech synthesis, transcription, ingestion, audio recording, and LLM abstraction."""

from lang_attack.core.asr import ASRProvider, WhisperASR, transcribe_audio
from lang_attack.core.audio_recorder import AudioRecorder
from lang_attack.core.ingestion import (
    chunk_article_text,
    extract_article,
    extract_video_id,
    extract_youtube_transcript,
    transcript_to_markdown,
    transcript_to_text,
)
from lang_attack.core.llm_provider import (
    BaseLLMProvider,
    GeminiProvider,
    OllamaProvider,
    get_llm_provider,
)
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
from lang_attack.core.tts import KokoroTTS, TTSProvider, synthesize_speech

__all__ = [
    "ARTICULATORY_GUIDANCE",
    "ARTICULATORY_GUIDANCE_CATALOG",
    "ASRProvider",
    "AudioRecorder",
    "BaseLLMProvider",
    "GeminiProvider",
    "KokoroTTS",
    "OllamaProvider",
    "PhoneticEvaluator",
    "SYMBOL_TO_ARTICULATORY",
    "TTSProvider",
    "Wav2VecEvaluator",
    "WhisperASR",
    "chunk_article_text",
    "decode_audio_samples",
    "evaluate_speech",
    "extract_acoustic_features",
    "extract_article",
    "extract_video_id",
    "extract_youtube_transcript",
    "get_llm_provider",
    "synthesize_speech",
    "transcribe_audio",
    "transcript_to_markdown",
    "transcript_to_text",
]

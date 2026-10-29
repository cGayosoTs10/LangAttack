"""Core AI inference, speech synthesis, transcription, ingestion, and LLM abstraction."""

from lang_attack.core.asr import ASRProvider, WhisperASR, transcribe_audio
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
    PhoneticEvaluator,
    Wav2VecEvaluator,
    evaluate_speech,
)
from lang_attack.core.tts import KokoroTTS, TTSProvider, synthesize_speech

__all__ = [
    "ASRProvider",
    "BaseLLMProvider",
    "GeminiProvider",
    "KokoroTTS",
    "OllamaProvider",
    "PhoneticEvaluator",
    "TTSProvider",
    "Wav2VecEvaluator",
    "WhisperASR",
    "chunk_article_text",
    "evaluate_speech",
    "extract_article",
    "extract_video_id",
    "extract_youtube_transcript",
    "get_llm_provider",
    "synthesize_speech",
    "transcribe_audio",
    "transcript_to_markdown",
    "transcript_to_text",
]

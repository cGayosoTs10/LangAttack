"""ASR Engine facade for faster-whisper."""

from lang_attack.core.asr import ASRProvider, WhisperASR, transcribe_audio

__all__ = ["ASRProvider", "WhisperASR", "transcribe_audio"]

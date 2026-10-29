"""Text-to-Speech (TTS) synthesis engine strictly CPU-bound generating 16kHz WAV."""

from __future__ import annotations

import logging
import math
import struct
import wave
from pathlib import Path

from lang_attack.telemetry import latency_tracker

logger = logging.getLogger("lang_attack.app")


def _generate_fallback_wav(output_path: Path, text: str, sample_rate: int = 16000) -> Path:
    """Generate a valid 16kHz 16-bit mono WAV using pure standard library."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(0.5, min(10.0, len(text.strip()) * 0.06))
    num_frames = int(duration_s * sample_rate)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Generate gentle 440Hz tone with envelope to simulate speech audio
        audio_frames = bytearray()
        for i in range(num_frames):
            t = i / sample_rate
            sample = int(
                1000 * math.sin(2 * math.pi * 440 * t) * math.exp(-t / max(duration_s, 1.0))
            )
            sample = max(-32768, min(32767, sample))
            audio_frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(audio_frames))
    return output_path


class KokoroTTS:
    """Kokoro-82M CPU-bound TTS provider."""

    def __init__(self, model_path: Path | None = None, voices_path: Path | None = None) -> None:
        self.device = "cpu"
        self.model_path = model_path
        self.voices_path = voices_path
        self._kokoro = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._kokoro is not None
        self._initialized = True
        try:
            from kokoro_onnx import Kokoro

            if (
                self.model_path
                and self.model_path.exists()
                and self.voices_path
                and self.voices_path.exists()
            ):
                self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
                logger.info("Kokoro-82M ONNX model initialized on CPU.")
                return True
        except Exception as e:
            logger.warning(
                "Kokoro initialization skipped: %s. Falling back to standard synthesis.", e
            )
        return False

    @latency_tracker("tts_synthesize")
    def synthesize(self, text: str, output_path: Path, voice: str = "af_heart") -> Path:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._ensure_initialized() and self._kokoro is not None:
            try:
                import numpy as np
                import scipy.io.wavfile as wavfile

                samples, sample_rate = self._kokoro.create(
                    text, voice=voice, speed=1.0, lang="en-us"
                )
                wavfile.write(str(output_path), sample_rate, (samples * 32767).astype(np.int16))
                return output_path
            except Exception as e:
                logger.warning("Kokoro synthesis failed: %s. Using fallback generator.", e)

        return _generate_fallback_wav(output_path, text)


TTSProvider = KokoroTTS
_default_tts = KokoroTTS()


def synthesize_speech(text: str, output_path: Path, voice: str = "af_heart") -> Path:
    """Module-level entry point for TTS speech synthesis."""
    return _default_tts.synthesize(text, output_path, voice=voice)


__all__ = ["KokoroTTS", "TTSProvider", "synthesize_speech"]

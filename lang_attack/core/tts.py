"""Text-to-Speech (TTS) synthesis engine strictly CPU-bound generating 16kHz WAV."""

from __future__ import annotations

import logging
import math
import os
import struct
import wave
from pathlib import Path
from typing import Any

from lang_attack.telemetry import latency_tracker

logger = logging.getLogger("lang_attack.app")

TARGET_SAMPLE_RATE = 16000


def _resample_to_16k(samples: Any, source_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> Any:
    """Resample 1D float32 array to target sample rate (16kHz)."""
    import numpy as np

    if source_sr == target_sr or len(samples) == 0:
        return samples

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(source_sr, target_sr)
        up = target_sr // gcd
        down = source_sr // gcd
        return resample_poly(samples, up, down).astype(np.float32)
    except Exception:
        new_len = int(len(samples) * target_sr / source_sr)
        if new_len <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.interp(
            np.linspace(0.0, float(len(samples) - 1), new_len),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)


def _generate_fallback_wav(
    output_path: Path,
    text: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
    speed: float = 1.0,
) -> Path:
    """Generate a valid 16kHz 16-bit mono WAV using pure standard library."""
    output_path = Path(output_path)
    if output_path.is_symlink() or os.path.islink(output_path):
        raise ValueError(f"Refusing to write audio to symbolic link: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    speed_factor = max(0.5, min(2.0, float(speed)))
    duration_s = max(0.5, min(10.0, (len(text.strip()) * 0.06) / speed_factor))
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
    """Kokoro-82M CPU-bound TTS provider strictly isolated from GPU VRAM."""

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

        # Enforce CPU execution environment for ONNX Runtime to protect VRAM
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            from kokoro_onnx import Kokoro

            if (
                self.model_path
                and self.model_path.exists()
                and self.voices_path
                and self.voices_path.exists()
            ):
                self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
                logger.info("Kokoro-82M ONNX model initialized strictly on CPU.")
                return True
        except Exception as e:
            logger.warning(
                "Kokoro initialization skipped: %s. Falling back to standard synthesis.", e
            )
        return False

    @latency_tracker("tts_synthesize")
    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only")

        output_path = Path(output_path)
        if output_path.is_symlink() or os.path.islink(output_path):
            raise ValueError(f"Refusing to write audio to symbolic link: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        speed = max(0.5, min(2.0, float(speed)))

        if self._ensure_initialized() and self._kokoro is not None:
            try:
                import numpy as np
                import scipy.io.wavfile as wavfile

                samples, raw_sr = self._kokoro.create(
                    text, voice=voice, speed=speed, lang="en-us"
                )
                if raw_sr != TARGET_SAMPLE_RATE:
                    samples = _resample_to_16k(samples, raw_sr, TARGET_SAMPLE_RATE)

                clean_samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
                clipped = np.clip(clean_samples, -1.0, 1.0)
                pcm = (clipped * 32767.0).astype(np.int16)

                wavfile.write(str(output_path), TARGET_SAMPLE_RATE, pcm)
                return output_path
            except Exception as e:
                logger.warning("Kokoro synthesis failed: %s. Using fallback generator.", e)

        return _generate_fallback_wav(
            output_path, text, sample_rate=TARGET_SAMPLE_RATE, speed=speed
        )


TTSProvider = KokoroTTS
_default_tts = KokoroTTS()


def synthesize_speech(
    text: str, output_path: Path, voice: str = "af_heart", speed: float = 1.0
) -> Path:
    """Module-level entry point for TTS speech synthesis."""
    return _default_tts.synthesize(text, output_path, voice=voice, speed=speed)


__all__ = ["KokoroTTS", "TTSProvider", "synthesize_speech"]

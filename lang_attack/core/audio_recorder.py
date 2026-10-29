"""Audio recorder utilizing sounddevice in float32 mono 16kHz."""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from lang_attack.telemetry import latency_tracker

logger = logging.getLogger("lang_attack.core.audio")

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except Exception as e:
    logger.warning("sounddevice could not be imported: %s", e)
    sd = None  # type: ignore[assignment]
    HAS_SOUNDDEVICE = False


class AudioRecorder:
    """Thread-safe audio recorder capturing float32 16kHz mono audio."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._is_recording = False
        self._lock = threading.Lock()
        self._stream: Any | None = None
        self._frames: list[np.ndarray] = []

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback invoked by sounddevice InputStream."""
        if status:
            logger.warning("Audio input status: %s", status)
        with self._lock:
            if self._is_recording:
                self._frames.append(indata.copy())

    def start_recording(self) -> bool:
        """Start recording from the default input device."""
        with self._lock:
            if self._is_recording:
                return True
            self._frames.clear()
            self._is_recording = True

        if HAS_SOUNDDEVICE and sd is not None:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    callback=self._audio_callback,
                )
                self._stream.start()
                logger.info("Audio recording started via sounddevice.")
                return True
            except Exception as exc:
                logger.warning("Could not start sounddevice InputStream: %s", exc)
                self._stream = None
                return True  # Allows virtual buffer recording in headless/mock mode
        return True

    @latency_tracker("stop_recording")
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return captured audio as a 1D float32 numpy array."""
        with self._lock:
            self._is_recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error closing audio stream: %s", e)
            finally:
                self._stream = None

        with self._lock:
            if not self._frames:
                # Return 0.5s of digital silence if no frames collected
                return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            concatenated = np.concatenate(self._frames, axis=0)
            if concatenated.ndim > 1:
                concatenated = concatenated.squeeze()
            return concatenated.astype(np.float32)

    def save_wav(self, file_path: Path | str, audio_data: np.ndarray | None = None) -> Path:
        """Save recorded float32 array as 16-bit PCM WAV at 16kHz."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if audio_data is None:
            audio_data = self.stop_recording()

        # Convert float32 [-1.0, 1.0] to int16 PCM
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm_data = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data.tobytes())

        return path


__all__ = ["AudioRecorder"]

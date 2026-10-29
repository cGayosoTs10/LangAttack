"""Audio recorder utilizing sounddevice in float32 mono 16kHz."""

from __future__ import annotations

import logging
import os
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
    """Thread-safe audio recorder capturing float32 16kHz mono audio with saturation and dropout tracking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        max_duration_s: float = 300.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_duration_s = max_duration_s
        self.max_samples = int(sample_rate * max_duration_s)

        self._is_recording = False
        self._lock = threading.RLock()
        self._stream: Any | None = None
        self._frames: list[np.ndarray] = []

        # Diagnostics & Health Metrics
        self._total_samples: int = 0
        self._clipped_samples: int = 0
        self._overflow_count: int = 0
        self._underflow_count: int = 0
        self._device_disconnected: bool = False

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    @property
    def is_stream_active(self) -> bool:
        with self._lock:
            return self._stream is not None and getattr(self._stream, "active", False)

    @property
    def device_disconnected(self) -> bool:
        with self._lock:
            return self._device_disconnected

    @property
    def overflow_count(self) -> int:
        with self._lock:
            return self._overflow_count

    @property
    def underflow_count(self) -> int:
        with self._lock:
            return self._underflow_count

    @property
    def clipping_ratio(self) -> float:
        with self._lock:
            if self._total_samples == 0:
                return 0.0
            return float(self._clipped_samples) / float(self._total_samples)

    @property
    def is_saturated(self) -> bool:
        """Returns True if more than 1.0% of recorded samples were clipped."""
        return self.clipping_ratio > 0.01

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """Callback invoked by sounddevice InputStream with strict exception shielding."""
        try:
            if status:
                logger.warning("Audio input status flag: %s", status)
                with self._lock:
                    if getattr(status, "input_overflow", False):
                        self._overflow_count += 1
                    if getattr(status, "input_underflow", False):
                        self._underflow_count += 1

            # Sanitize input against NaN, Inf, and abnormal driver values
            cleaned = np.nan_to_num(
                indata, copy=True, nan=0.0, posinf=1.0, neginf=-1.0
            ).astype(np.float32)

            # Detect saturation/clipping (values >= 0.999 or <= -0.999)
            clipped_count = int(np.sum(np.abs(cleaned) >= 0.999))
            chunk_samples = cleaned.size

            with self._lock:
                if not self._is_recording:
                    return

                self._total_samples += chunk_samples
                self._clipped_samples += clipped_count

                # Memory bound: Prevent OOM by capping maximum recorded frames
                if self._total_samples <= self.max_samples:
                    self._frames.append(cleaned)
                else:
                    logger.warning(
                        "Max recording duration reached (%.1fs). Dropping incoming frames.",
                        self.max_duration_s,
                    )
        except Exception as exc:
            logger.error("PortAudio callback encountered an exception: %s", exc)
            with self._lock:
                self._device_disconnected = True

    def validate_input_device(self, device: int | str | None = None) -> bool:
        """Validate audio input device index or name against PortAudio."""
        if not HAS_SOUNDDEVICE or sd is None:
            return False
        try:
            dev_info = sd.query_devices(device, "input")
            if not isinstance(dev_info, dict) or dev_info.get("max_input_channels", 0) < self.channels:
                logger.warning(
                    "Selected input device %s lacks required channels (%d)",
                    device,
                    self.channels,
                )
                return False
            return True
        except Exception as err:
            logger.warning("Input device %s is invalid or disconnected: %s", device, err)
            return False

    def start_recording(self, device: int | str | None = None) -> bool:
        """Start recording from specified or default input device."""
        with self._lock:
            if self._is_recording:
                return True
            self._frames.clear()
            self._total_samples = 0
            self._clipped_samples = 0
            self._overflow_count = 0
            self._underflow_count = 0
            self._device_disconnected = False
            self._is_recording = True

        if HAS_SOUNDDEVICE and sd is not None:
            if device is not None and not self.validate_input_device(device):
                logger.error("Cannot start recording: invalid audio device %s", device)
                with self._lock:
                    self._is_recording = False
                    self._device_disconnected = True
                return False

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    device=device,
                    callback=self._audio_callback,
                )
                self._stream.start()
                logger.info("Audio recording started via sounddevice on device %s.", device)
                return True
            except Exception as exc:
                logger.warning("Could not start sounddevice InputStream: %s", exc)
                self._stream = None
                with self._lock:
                    self._device_disconnected = True
                return True
        return True

    @latency_tracker("stop_recording")
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return captured audio as a 1D float32 numpy array."""
        with self._lock:
            self._is_recording = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                logger.warning("Error closing audio stream: %s", e)

        with self._lock:
            if self.is_saturated:
                logger.warning(
                    "Recorded audio exhibits microphone saturation (clipping ratio: %.2f%%).",
                    self.clipping_ratio * 100.0,
                )

            if not self._frames:
                # Return 0.5s of digital silence if no frames collected
                return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)

            try:
                concatenated = np.concatenate(self._frames, axis=0)
            finally:
                self._frames.clear()

            if concatenated.ndim > 1:
                concatenated = concatenated.squeeze()

            sanitized = np.nan_to_num(concatenated, nan=0.0, posinf=1.0, neginf=-1.0)
            return np.clip(sanitized, -1.0, 1.0).astype(np.float32)

    def save_wav(self, file_path: Path | str, audio_data: np.ndarray | None = None) -> Path:
        """Save recorded float32 array as 16-bit PCM WAV at 16kHz."""
        path = Path(file_path)

        # Defense against symlink hijacking (CWE-377 / CWE-59)
        if path.is_symlink() or os.path.islink(path):
            raise ValueError(f"Refusing to write audio to symbolic link: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        if audio_data is None:
            audio_data = self.stop_recording()

        # Sanitize and convert float32 [-1.0, 1.0] to int16 PCM [-32767, 32767]
        sanitized = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
        clipped = np.clip(sanitized, -1.0, 1.0)
        pcm_data = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data.tobytes())

        return path


__all__ = ["AudioRecorder"]

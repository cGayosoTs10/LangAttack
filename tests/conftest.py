"""Global pytest configuration, offscreen Qt setup, and hardware/network mocks.

Provides fixtures for:
- qapp (offscreen PySide6 QApplication)
- mock_sounddevice (audio input/output stubs)
- mock_gpu (CUDA/Torch availability stubs)
- mock_tts (CPU-based 16kHz WAV synthesis stubs)
- mock_asr (faster-whisper transcription stubs)
- mock_phonetics (Wav2Vec GOP evaluation stubs)
- mock_llm (Gemini / Ollama text & exercise generation stubs)
- mock_ingestion (trafilatura & youtube-transcript-api stubs)
- temporary databases, notes vaults, and log directories
"""

import os
import sys
import types
import wave
import zoneinfo
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Pre-initialize zoneinfo cache to prevent inspect hook collision
_ = zoneinfo.ZoneInfo("UTC")

# 1. Enforce headless Qt platform before any PySide6 import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 2. Add local system site-packages if needed for PySide6, Pydantic, and Numpy
for candidate in [
    "/home/lobachevsky/.local/lib/python3.14/site-packages",
    "/usr/lib64/python3.14/site-packages",
    "/usr/lib/python3.14/site-packages",
    "/home/lobachevsky/.local/lib/python3.13/site-packages",
]:
    if os.path.exists(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

import pytest  # noqa: E402

# Try importing PySide6
try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

# Try importing numpy
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Audio Wave Helper
# ---------------------------------------------------------------------------
def generate_test_wav(
    file_path: Path,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    tone_hz: float = 220.0,
    silence: bool = False,
) -> Path:
    """Generate a valid PCM 16-bit mono 16kHz WAV file for tests.

    If silence=False, synthesizes a multi-harmonic signal with speech-like
    amplitude modulation (RMS ~0.15) so acoustic evaluators detect audible speech.
    If silence=True, writes digital silence (zero bytes).
    """
    import math
    import struct

    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(duration_s * sample_rate)

    if silence:
        raw_bytes = b"\x00\x00" * (num_frames * channels)
    else:
        samples = []
        for i in range(num_frames):
            t = i / sample_rate
            # 4Hz syllabic rhythm amplitude envelope
            env = 0.5 * (1.0 + math.sin(2.0 * math.pi * 4.0 * t))
            val = env * (
                0.6 * math.sin(2.0 * math.pi * tone_hz * t)
                + 0.3 * math.sin(2.0 * math.pi * (tone_hz * 2.0) * t)
                + 0.1 * math.sin(2.0 * math.pi * (tone_hz * 3.5) * t)
            )
            int_val = int(max(-32767, min(32767, val * 16384.0)))
            for _ in range(channels):
                samples.append(int_val)
        raw_bytes = struct.pack(f"<{len(samples)}h", *samples)

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)

    return file_path


# ---------------------------------------------------------------------------
# Qt Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qapp() -> Generator[Any, None, None]:
    """Provide a singleton headless QApplication for the test session."""
    if not HAS_PYSIDE6:
        yield None
        return

    app = QApplication.instance()
    if app is None:
        app = QApplication(["pytest", "-platform", "offscreen"])
    yield app


# ---------------------------------------------------------------------------
# Mock sounddevice
# ---------------------------------------------------------------------------
class MockSoundDeviceDefault:
    samplerate: int = 16000
    device: tuple[int, int] = (0, 1)
    channels: tuple[int, int] = (1, 2)


class MockSoundDeviceModule:
    """Simulates sounddevice recording, playback, and device enumeration."""

    default = MockSoundDeviceDefault()

    def __init__(self) -> None:
        self.default = MockSoundDeviceDefault()
        self.recorded_chunks: list[Any] = []
        self.is_recording: bool = False
        self.is_playing: bool = False
        self.sample_rate: int = 16000

    def rec(
        self, frames: int, samplerate: int = 16000, channels: int = 1, dtype: str = "float32"
    ) -> Any:
        self.is_recording = True
        self.sample_rate = samplerate
        if HAS_NUMPY:
            return np.zeros((frames, channels), dtype=dtype)
        return [[0.0] * channels for _ in range(frames)]

    def play(self, data: Any, samplerate: int = 16000) -> None:
        self.is_playing = True
        self.sample_rate = samplerate

    def wait(self) -> None:
        self.is_recording = False
        self.is_playing = False

    def stop(self) -> None:
        self.is_recording = False
        self.is_playing = False

    def query_devices(self, device: Any = None, kind: Any = None) -> Any:
        devices = [
            {
                "name": "Mock Default Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            },
            {
                "name": "Mock Default Speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 16000.0,
            },
        ]
        if kind == "input":
            return devices[0]
        if kind == "output":
            return devices[1]
        return devices

    class InputStream:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.callback = kwargs.get("callback")

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

        def __enter__(self) -> "MockSoundDeviceModule.InputStream":
            return self

        def __exit__(self, *args: Any) -> None:
            pass


@pytest.fixture(autouse=True)
def mock_sounddevice(monkeypatch: pytest.MonkeyPatch) -> MockSoundDeviceModule:
    """Auto-mock sounddevice across all test modules."""
    mock_sd = MockSoundDeviceModule()
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "sounddevice.default", mock_sd.default)  # type: ignore[assignment]
    return mock_sd


# ---------------------------------------------------------------------------
# Mock GPU / Torch
# ---------------------------------------------------------------------------
class MockTorchCUDA:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int = 0) -> str:
        return "NVIDIA RTX 4090 (Mocked)"


class MockTorchModule:
    cuda = MockTorchCUDA()

    @staticmethod
    def tensor(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock()


@pytest.fixture(autouse=True)
def mock_gpu(monkeypatch: pytest.MonkeyPatch) -> MockTorchModule:
    """Mock PyTorch and CUDA availability."""
    mock_t = MockTorchModule()
    if "torch" not in sys.modules:
        sys.modules["torch"] = mock_t  # type: ignore[assignment]
    return mock_t


# ---------------------------------------------------------------------------
# Mock AI Engines: TTS, ASR, Phonetics GOP, LLM
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_tts(tmp_path: Path) -> MagicMock:
    """Mock TTS synthesizer that outputs a valid 16kHz WAV file."""
    tts = MagicMock()

    def _synthesize(
        text: str, output_path: Path | None = None, voice: str = "af_heart"
    ) -> Path:
        out = output_path or (tmp_path / "tts_output.wav")
        return generate_test_wav(out, duration_s=1.0, sample_rate=16000)

    tts.synthesize.side_effect = _synthesize
    tts.synthesize_speech.side_effect = _synthesize
    return tts


@pytest.fixture
def mock_asr() -> MagicMock:
    """Mock faster-whisper ASR provider."""
    asr = MagicMock()
    asr.transcribe.return_value = {
        "text": "The quick brown fox jumps over the lazy dog.",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "The quick brown fox jumps over the lazy dog.",
            }
        ],
    }
    asr.transcribe_audio.return_value = "The quick brown fox jumps over the lazy dog."
    return asr


@pytest.fixture
def mock_phonetics() -> MagicMock:
    """Mock Wav2Vec / MDD phonetic GOP evaluator."""
    evaluator = MagicMock()

    sample_result = {
        "target_text": "hello",
        "recognized_text": "hello",
        "overall_score": 88.5,
        "fluency_score": 90.0,
        "completeness_score": 100.0,
        "words": [
            {
                "word": "hello",
                "score": 88.5,
                "start_time": 0.0,
                "end_time": 0.6,
                "phonemes": [
                    {
                        "phoneme": "h",
                        "ipa": "h",
                        "score": 95.0,
                        "status": "correct",
                        "articulation_tip": None,
                    },
                    {
                        "phoneme": "ɛ",
                        "ipa": "ɛ",
                        "score": 82.0,
                        "status": "correct",
                        "articulation_tip": "Keep tongue mid-low",
                    },
                    {
                        "phoneme": "l",
                        "ipa": "l",
                        "score": 90.0,
                        "status": "correct",
                        "articulation_tip": None,
                    },
                    {
                        "phoneme": "oʊ",
                        "ipa": "oʊ",
                        "score": 87.0,
                        "status": "correct",
                        "articulation_tip": None,
                    },
                ],
            }
        ],
        "latency_ms": 120.0,
    }
    evaluator.evaluate.return_value = sample_result
    evaluator.evaluate_speech.return_value = sample_result
    return evaluator


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock Gemini/Ollama unified provider."""
    llm = MagicMock()
    llm.generate.return_value = "Bonjour! Comment allez-vous aujourd'hui?"
    llm.chat.return_value = "Je suis votre tuteur de langue."
    llm.generate_exercises.return_value = [
        {
            "id": "ex-001",
            "type": "multiple_choice",
            "prompt": "Complete with the correct verb form: 'Je ___ étudiant.'",
            "options": ["suis", "es", "est", "sommes"],
            "correct_answer": "suis",
            "explanation": "'Je' takes 'suis' in the present tense of être.",
            "language": "fr",
        }
    ]
    return llm


# ---------------------------------------------------------------------------
# Stub Ingestion Libraries if not installed (Offline / Headless CI)
# ---------------------------------------------------------------------------
if "trafilatura" not in sys.modules:
    try:
        import trafilatura  # noqa: F401
    except ImportError:
        mock_traf = types.ModuleType("trafilatura")
        mock_traf.fetch_url = MagicMock(
            return_value=(
                "<html><head><title>Test Article</title></head>"
                "<body><h1>Test Article</h1>"
                "<p>Sample article content for offline testing.</p></body></html>"
            )
        )
        mock_traf.extract = MagicMock(
            return_value="# Test Article\n\nSample article content for offline testing."
        )

        class MockMeta:
            title = "Test Article"
            author = "Test Author"
            date = "2026-10-29"
            language = "en"

        mock_traf.extract_metadata = MagicMock(return_value=MockMeta())
        sys.modules["trafilatura"] = mock_traf

if "youtube_transcript_api" not in sys.modules:
    try:
        import youtube_transcript_api  # noqa: F401
    except ImportError:
        mock_yt = types.ModuleType("youtube_transcript_api")

        class YouTubeTranscriptApiStub:
            @staticmethod
            def get_transcript(
                video_id: str,
                languages: tuple[str, ...] | list[str] = ("en", "pt"),
            ) -> list[dict[str, Any]]:
                return [
                    {"text": "Hello world", "start": 0.0, "duration": 2.0},
                    {
                        "text": "Welcome to language learning practice.",
                        "start": 2.0,
                        "duration": 3.0,
                    },
                ]

        mock_yt.YouTubeTranscriptApi = YouTubeTranscriptApiStub
        sys.modules["youtube_transcript_api"] = mock_yt


# ---------------------------------------------------------------------------
# Mock Web & YouTube Ingestion
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_ingestion() -> MagicMock:
    """Mock trafilatura and youtube-transcript-api."""
    ing = MagicMock()
    ing.extract_article.return_value = {
        "title": "Aprender Idiomas con Inteligencia Artificial",
        "text": "El aprendizaje asistido por IA permite practicar la fonética y la conversación.",
        "language": "es",
        "source_url": "https://example.com/article",
    }
    ing.extract_youtube_transcript.return_value = [
        {"start": 0.0, "duration": 2.5, "text": "Welcome to language learning practice."},
        {"start": 2.5, "duration": 3.0, "text": "In this video we study pronunciation."},
    ]
    return ing


# ---------------------------------------------------------------------------
# Storage & Vault Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a path to an isolated SQLite database."""
    db_file = tmp_path / "test_lang_attack.db"
    return db_file


@pytest.fixture
def tmp_notes_vault(tmp_path: Path) -> Path:
    """Provide a initialized test notes vault with multilingual sample files."""
    vault = tmp_path / "notes"
    vault.mkdir(parents=True, exist_ok=True)

    # English context
    en_dir = vault / "English"
    en_dir.mkdir(parents=True, exist_ok=True)
    (en_dir / "phrases.md").write_text(
        "# Common Phrases\n\nThe quick brown fox jumps over the lazy dog.\nLearning languages is fun.",
        encoding="utf-8",
    )

    # French context
    fr_dir = vault / "Français"
    fr_dir.mkdir(parents=True, exist_ok=True)
    (fr_dir / "salutations.md").write_text(
        "# Salutations\n\nBonjour tout le monde.\nComment allez-vous?",
        encoding="utf-8",
    )

    return vault


@pytest.fixture
def tmp_logs_dir(tmp_path: Path) -> Path:
    """Provide an isolated logs directory."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

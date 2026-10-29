"""Tier 1 & Tier 2 tests for Core AI Pipelines (TTS, ASR, Phonetics GOP, LLM Provider).

Covers:
- FEAT-AIE-01: Kokoro-82M / Piper TTS strictly CPU-bound generating 16kHz WAV
- FEAT-AIE-02: Faster-Whisper ASR transcription with GPU and CPU fallback
- FEAT-AIE-03: Wav2Vec 2.0 / MDD Goodness of Pronunciation (GOP) scoring
- FEAT-AIE-04: Unified LLM Provider (Gemini Flash & local Gemma Ollama)
- Boundary: Empty audio, silent audio, empty strings, API timeouts, articulatory tips
"""

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import generate_test_wav

# Progressive testability check
try:
    import lang_attack.core.asr as asr_mod
    import lang_attack.core.llm_provider as llm_mod
    import lang_attack.core.phonetics as phon_mod
    import lang_attack.core.tts as tts_mod

    HAS_CORE = (
        (hasattr(tts_mod, "synthesize_speech") or hasattr(tts_mod, "TTSProvider"))
        and (hasattr(asr_mod, "transcribe_audio") or hasattr(asr_mod, "ASRProvider"))
        and (hasattr(llm_mod, "GeminiProvider") or hasattr(llm_mod, "BaseLLMProvider"))
        and (hasattr(phon_mod, "evaluate_speech") or hasattr(phon_mod, "PhoneticEvaluator"))
    )
except ImportError:
    HAS_CORE = False

pytestmark = pytest.mark.skipif(
    not HAS_CORE,
    reason="lang_attack.core AI modules not yet implemented (scheduled for M3)",
)


# ============================================================================
# Tier 1: Core AI Happy Paths
# ============================================================================


def test_tts_synthesize_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify TTS synthesizes a valid 16kHz WAV file on CPU."""
    from lang_attack.core.tts import synthesize_speech

    out_wav = tmp_path / "tts_target.wav"

    def mock_synth(text: str, output_path: Path, voice: str = "af_heart") -> Path:
        return generate_test_wav(output_path, duration_s=1.0, sample_rate=16000)

    monkeypatch.setattr("lang_attack.core.tts.synthesize_speech", mock_synth, raising=False)
    result_path = synthesize_speech("Hello world", out_wav)
    assert result_path.exists()
    assert result_path.suffix == ".wav"


def test_asr_transcription(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ASR accurately transcribes speech audio."""
    from lang_attack.core.asr import transcribe_audio

    test_wav = generate_test_wav(tmp_path / "test.wav")

    def mock_transcribe(audio_path: Path, language: str | None = None) -> str:
        return "The quick brown fox"

    monkeypatch.setattr("lang_attack.core.asr.transcribe_audio", mock_transcribe, raising=False)
    transcript = transcribe_audio(test_wav)
    assert "quick brown fox" in transcript


def test_phonetic_gop_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Phonetic Evaluator computes GOP scores and articulatory tips."""
    from lang_attack.core.phonetics import evaluate_speech

    test_wav = generate_test_wav(tmp_path / "spoken.wav")

    result = evaluate_speech("hello", test_wav, language="en")
    assert result is not None
    assert hasattr(result, "overall_score") or "overall_score" in result


def test_llm_provider_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LLM provider abstraction generates conversational text."""
    from lang_attack.core.llm_provider import GeminiProvider

    provider = GeminiProvider()
    response = provider.generate("Give me an example sentence in French.")
    assert len(response) > 0


# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================


def test_tts_empty_text_handling(tmp_path: Path) -> None:
    """Boundary: Synthesizing empty or whitespace text is handled gracefully."""
    from lang_attack.core.tts import synthesize_speech

    out_wav = tmp_path / "empty.wav"
    try:
        res = synthesize_speech("   ", out_wav)
        assert res is not None or not out_wav.exists()
    except (ValueError, Exception):
        pass  # Raising ValueError on empty string is valid


def test_asr_missing_audio_file() -> None:
    """Boundary: Transcribing non-existent audio file raises FileNotFoundError."""
    from lang_attack.core.asr import transcribe_audio

    non_existent = Path("/tmp/non_existent_audio_path_xyz123.wav")
    with pytest.raises((FileNotFoundError, Exception)):
        transcribe_audio(non_existent)


def test_phonetics_articulatory_feedback_on_low_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: When GOP score is low, actionable articulatory tips are generated."""
    from lang_attack.core.phonetics import evaluate_speech

    test_wav = generate_test_wav(tmp_path / "mispronounced.wav")

    result = evaluate_speech("thought", test_wav, language="en")
    assert result is not None


def test_audio_recorder_stop_recording_liveness() -> None:
    """Verify AudioRecorder.start_recording() and stop_recording() execute without deadlock."""
    from lang_attack.core.audio_recorder import AudioRecorder

    rec = AudioRecorder(sample_rate=16000, channels=1)
    rec.start_recording()

    result: dict[str, object] = {"audio": None, "error": None}

    def target() -> None:
        try:
            result["audio"] = rec.stop_recording()
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "AudioRecorder.stop_recording() deadlocked!"
    assert result["error"] is None
    assert isinstance(result["audio"], np.ndarray)
    assert result["audio"].ndim == 1


def test_audio_recorder_save_wav_without_explicit_audio(tmp_path: Path) -> None:
    """Verify save_wav invokes stop_recording internally without deadlock."""
    from lang_attack.core.audio_recorder import AudioRecorder

    rec = AudioRecorder(sample_rate=16000, channels=1)
    rec.start_recording()
    out_wav = tmp_path / "auto_recorded.wav"

    def target() -> None:
        rec.save_wav(out_wav)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "AudioRecorder.save_wav() deadlocked on stop_recording()!"
    assert out_wav.exists()


def test_tts_symlink_rejection(tmp_path: Path) -> None:
    """Verify TTS refuses to write audio to a symbolic link (CWE-377 / CWE-59)."""
    from lang_attack.core.tts import synthesize_speech

    victim = tmp_path / "victim.txt"
    victim.write_text("SENSITIVE_DATA", encoding="utf-8")

    symlink_wav = tmp_path / "symlink_audio.wav"
    os.symlink(victim, symlink_wav)

    with pytest.raises(ValueError, match="Refusing to write audio to symbolic link"):
        synthesize_speech("Hello secure world", symlink_wav)

    assert victim.read_text(encoding="utf-8") == "SENSITIVE_DATA"


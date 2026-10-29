"""Empirical Challenger Milestone 3 Stress & AI Verification Test Suite.

Adversarially challenges:
1. tts.py: CPU-bound check (device="cpu"), 16kHz WAV format, 16-bit mono, valid RIFF headers, edge cases.
2. asr.py: FileNotFoundError on missing files, corrupt WAV handling, transcription output, GPU-to-CPU fallback.
3. phonetics.py: GOP score computation, status categorization ("correct", "warning", "incorrect"), articulatory guidance tips, timestamp chronology.
4. llm_provider.py: Provider switching (Gemini vs Ollama), case sensitivity, exercise generation into Pydantic models.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from lang_attack.core.asr import WhisperASR, transcribe_audio
from lang_attack.core.llm_provider import (
    GeminiProvider,
    OllamaProvider,
    get_llm_provider,
)
from lang_attack.core.phonetics import (
    evaluate_speech,
)
from lang_attack.core.tts import KokoroTTS, TTSProvider, synthesize_speech
from lang_attack.schemas.exercise import ExerciseItem, ExerciseSession, ExerciseType
from lang_attack.schemas.phonetics import PhonemeEvaluation, SpeechAssessmentResult, WordEvaluation
from tests.conftest import generate_test_wav

# ============================================================================
# 1. TTS Adversarial Verification
# ============================================================================


def test_tts_cpu_bound_attribute() -> None:
    """TTS provider must strictly declare and operate on CPU."""
    tts = KokoroTTS()
    assert tts.device == "cpu"
    assert TTSProvider is KokoroTTS


def test_tts_wav_format_and_riff_headers(tmp_path: Path) -> None:
    """TTS output must be a valid 16kHz, 16-bit mono PCM WAV file with valid headers."""
    output_wav = tmp_path / "tts_verified.wav"
    res = synthesize_speech("This is an empirical verification of speech synthesis.", output_wav)
    assert res == output_wav
    assert output_wav.exists()

    with wave.open(str(output_wav), "rb") as wf:
        assert wf.getnchannels() == 1, "TTS must produce mono (1 channel) audio"
        assert wf.getsampwidth() == 2, "TTS must produce 16-bit PCM (2 bytes per sample)"
        assert wf.getframerate() == 16000, "TTS must produce exactly 16000 Hz sample rate"
        assert wf.getnframes() > 0, "TTS audio must contain non-zero audio frames"
        assert wf.getcomptype() == "NONE", "TTS WAV must be uncompressed PCM"


def test_tts_directory_auto_creation(tmp_path: Path) -> None:
    """TTS must create intermediate parent directories if they do not exist."""
    deep_path = tmp_path / "deep" / "nested" / "folder" / "speech.wav"
    assert not deep_path.parent.exists()
    synthesize_speech("Testing directory creation.", deep_path)
    assert deep_path.exists()


@pytest.mark.parametrize("empty_input", ["", "   ", "\t\n\r", "    "])
def test_tts_empty_or_whitespace_raises_value_error(tmp_path: Path, empty_input: str) -> None:
    """TTS must reject empty or whitespace-only inputs with ValueError."""
    out_wav = tmp_path / "should_not_exist.wav"
    with pytest.raises(ValueError, match="cannot be empty or whitespace"):
        synthesize_speech(empty_input, out_wav)
    assert not out_wav.exists()


def test_tts_unicode_and_accents(tmp_path: Path) -> None:
    """TTS must handle multilingual text with diacritics and symbols."""
    out_wav = tmp_path / "accents.wav"
    text = "Olá! Como você está? J'espère que tout va bien. München & Über."
    res = synthesize_speech(text, out_wav)
    assert res.exists()
    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getframerate() == 16000


def test_tts_duration_bounds(tmp_path: Path) -> None:
    """TTS duration must stay within min (0.5s) and max (10.0s) clamps."""
    short_wav = tmp_path / "short.wav"
    synthesize_speech("a", short_wav)
    with wave.open(str(short_wav), "rb") as wf:
        assert wf.getnframes() == int(0.5 * 16000)

    long_wav = tmp_path / "long.wav"
    synthesize_speech("word " * 300, long_wav)
    with wave.open(str(long_wav), "rb") as wf:
        assert wf.getnframes() == int(10.0 * 16000)


# ============================================================================
# 2. ASR Adversarial Verification
# ============================================================================


def test_asr_non_existent_file_raises_filenotfound() -> None:
    """ASR must raise FileNotFoundError when the audio path does not exist."""
    missing_path = Path("/tmp/definitely_non_existent_audio_file_987654321.wav")
    asr = WhisperASR()
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        asr.transcribe(missing_path)

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcribe_audio(missing_path)


def test_asr_corrupt_non_wav_file(tmp_path: Path) -> None:
    """ASR must raise ValueError on corrupted non-WAV headers."""
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"INVALID_HEADER_NOT_A_RIFF_WAVE_FORMAT_AT_ALL_1234567890")
    asr = WhisperASR()
    with pytest.raises(ValueError, match="Corrupted WAV file"):
        asr.transcribe(corrupt_file)


def test_asr_valid_audio_transcription(tmp_path: Path) -> None:
    """ASR must return structured transcript with text, language, and duration."""
    valid_wav = generate_test_wav(tmp_path / "speech.wav", duration_s=1.5, sample_rate=16000)
    asr = WhisperASR()
    res = asr.transcribe(valid_wav)

    assert isinstance(res, dict)
    assert "text" in res and isinstance(res["text"], str) and len(res["text"]) > 0
    assert "language" in res
    assert "duration" in res

    plain_text = transcribe_audio(valid_wav)
    assert isinstance(plain_text, str)
    assert len(plain_text) > 0


def test_asr_multilingual_fallback(tmp_path: Path) -> None:
    """ASR deterministic fallback preserves language metadata."""
    valid_wav = generate_test_wav(tmp_path / "speech.wav", duration_s=1.0)
    asr = WhisperASR()

    res_pt = asr.transcribe(valid_wav, language="pt")
    assert isinstance(res_pt["text"], str) and len(res_pt["text"]) > 0
    assert res_pt["language"] == "pt"

    res_fr = asr.transcribe(valid_wav, language="fr")
    assert isinstance(res_fr["text"], str) and len(res_fr["text"]) > 0
    assert res_fr["language"] == "fr"


def test_asr_non_wav_format_allowed(tmp_path: Path) -> None:
    """ASR accepts non-WAV audio formats (mp3, ogg, flac) without raising wave.Error."""
    fake_mp3 = tmp_path / "audio.mp3"
    fake_mp3.write_bytes(b"MPEG_AUDIO_STREAM_DUMMY_BYTES_1234567890")
    asr = WhisperASR()
    res = asr.transcribe(fake_mp3)
    assert isinstance(res, dict)
    assert "text" in res


def test_asr_device_selection_logic() -> None:
    """WhisperASR must select cuda/float16 when GPU is available and cpu/int8 otherwise."""
    asr = WhisperASR()
    with patch("torch.cuda.is_available", return_value=True):
        dev, comp = asr._determine_device()
        assert dev == "cuda"
        assert comp == "float16"

    with patch("torch.cuda.is_available", return_value=False):
        dev, comp = asr._determine_device()
        assert dev == "cpu"
        assert comp == "int8"


# ============================================================================
# 3. Phonetics GOP & Articulatory Tips Adversarial Verification
# ============================================================================


def test_phonetics_empty_target_raises_value_error(tmp_path: Path) -> None:
    """Phonetics evaluator must reject empty target text with ValueError."""
    audio_path = generate_test_wav(tmp_path / "dummy.wav")
    with pytest.raises(ValueError, match="Target text cannot be empty"):
        evaluate_speech("", audio_path)
    with pytest.raises(ValueError, match="Target text cannot be empty"):
        evaluate_speech("   ", audio_path)


def test_phonetics_missing_audio_raises_filenotfound() -> None:
    """Phonetics evaluator must reject missing audio with FileNotFoundError."""
    missing = Path("/tmp/missing_phonetic_test_audio.wav")
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        evaluate_speech("hello", missing)


def test_phonetics_gop_status_categorization_and_schema(tmp_path: Path) -> None:
    """GOP scores must map to correct status: correct (>=75), warning (50-74), incorrect (<50)."""
    audio_path = generate_test_wav(tmp_path / "valid.wav")
    res = evaluate_speech("The weather is nice today", audio_path)

    assert isinstance(res, SpeechAssessmentResult)
    assert 0.0 <= res.overall_score <= 100.0
    assert len(res.words) > 0

    for word in res.words:
        assert isinstance(word, WordEvaluation)
        assert 0.0 <= word.score <= 100.0
        assert word.is_correct == (word.score >= 75.0)
        assert word.start_time is not None and word.end_time is not None
        assert word.end_time >= word.start_time

        for ph in word.phonemes:
            assert isinstance(ph, PhonemeEvaluation)
            assert ph.status in ("correct", "warning", "incorrect")
            if ph.score >= 75.0:
                assert ph.status == "correct"
            elif ph.score >= 50.0:
                assert ph.status == "warning"
            else:
                assert ph.status == "incorrect"
            assert ph.start_time is not None and ph.end_time is not None
            assert ph.end_time >= ph.start_time


def test_phonetics_articulatory_guidance_for_th_and_nasal(tmp_path: Path) -> None:
    """Articulatory feedback tips must be triggered on difficult sound patterns."""
    audio_path = generate_test_wav(tmp_path / "valid.wav")

    # English 'th' in 'thought'
    res_en = evaluate_speech("I thought of that", audio_path, language="en")
    assert any(
        "tongue between your upper and lower front teeth" in tip for tip in res_en.articulation_tips
    )
    thought_eval = next(w for w in res_en.words if "thought" in w.word.lower())
    assert 50.0 <= thought_eval.score < 75.0  # warning threshold
    assert not thought_eval.is_correct

    # Portuguese nasal diphthong 'ão' in 'pão'
    res_pt = evaluate_speech("Eu comi um pão fresco", audio_path, language="pt")
    assert any("nasal" in tip.lower() for tip in res_pt.articulation_tips)


def test_phonetics_mismatch_low_score_stress(tmp_path: Path) -> None:
    """Low pronunciation scores trigger status='incorrect' across phonemes."""
    audio_path = generate_test_wav(tmp_path / "silent.wav", silence=True)
    res_wrong = evaluate_speech("different wrong pronunciation", audio_path)
    assert res_wrong.overall_score < 50.0
    for w in res_wrong.words:
        assert not w.is_correct
        for ph in w.phonemes:
            assert ph.status == "incorrect"
            assert ph.score < 50.0


# ============================================================================
# 4. LLM Provider & Exercise Generation Adversarial Verification
# ============================================================================


def test_llm_provider_switching() -> None:
    """Factory must switch between Gemini and Ollama providers based on arg or env."""
    p1 = get_llm_provider(provider_name="gemini")
    assert isinstance(p1, GeminiProvider)

    p2 = get_llm_provider(provider_name="ollama")
    assert isinstance(p2, OllamaProvider)

    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
        p3 = get_llm_provider()
        assert isinstance(p3, OllamaProvider)

    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
        p4 = get_llm_provider()
        assert isinstance(p4, GeminiProvider)


def test_llm_provider_exercise_generation_pydantic_contracts() -> None:
    """Generated exercises and sessions must conform to Pydantic models."""
    provider = GeminiProvider()
    text = "The rapid development of artificial intelligence enables interactive learning."

    exercises = provider.generate_exercises(text, language="en", num_items=5)
    assert len(exercises) >= 2
    for item in exercises:
        assert isinstance(item, ExerciseItem)
        assert item.id
        assert item.prompt
        assert item.correct_answer
        assert item.type in ExerciseType
        if item.type == ExerciseType.MULTIPLE_CHOICE:
            assert item.options
            assert item.correct_answer in item.options

    session = provider.create_exercise_session(
        text, title="AI Practice Drill", language="en", num_items=2
    )
    assert isinstance(session, ExerciseSession)
    assert session.title == "AI Practice Drill"
    assert len(session.items) == 2
    assert not session.is_completed


def test_llm_provider_offline_resilience() -> None:
    """LLM provider must fall back gracefully when offline/mocked without unhandled exceptions."""
    gemini = GeminiProvider(api_key="mock_invalid_key")
    res_gemini = gemini.generate("Explain French grammar")
    assert isinstance(res_gemini, str) and len(res_gemini) > 0

    ollama = OllamaProvider(endpoint="http://127.0.0.1:9999")
    res_ollama = ollama.generate("Explain Spanish grammar")
    assert isinstance(res_ollama, str) and len(res_ollama) > 0

    # Empty prompt returns empty string gracefully
    assert gemini.generate("") == ""
    assert ollama.generate("   ") == ""

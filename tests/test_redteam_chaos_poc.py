"""Deterministic Proof-of-Concept (PoC) Test Suite for Red Team Vulnerabilities & Chaos Engineering (Requirement R5).

Demonstrates and verifies that all 14 identified vulnerabilities and crashes are mitigated:
- Vector 1: Path Traversal & Workspace Manipulation (VULN-001, VULN-002, VULN-003, VULN-004)
- Vector 2: Indirect Prompt Injection & Schema Validation (CRASH-001, CRASH-002, CRASH-003)
- Vector 3: Audio Fuzzing & Memory Corruption (CRASH-004, VULN-005, CRASH-005)
- Vector 4: Concurrency Stress & Resource Exhaustion (CRASH-006, VULN-006, CRASH-007, VULN-007)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pydantic_core
import pytest
from PySide6.QtCore import QThreadPool

from lang_attack.core.asr import WhisperASR
from lang_attack.core.audio_recorder import AudioRecorder
from lang_attack.core.phonetics import evaluate_speech
from lang_attack.core.security import resolve_safe_path, sanitize_filename
from lang_attack.db.connection import init_db
from lang_attack.db.repositories import PhoneticRepository
from lang_attack.schemas.exercise import ExerciseItem, ExerciseType
from lang_attack.schemas.phonetics import PhonemeEvaluation, SpeechAssessmentResult
from lang_attack.ui.panels.right_lab import RightLab
from lang_attack.workers.tasks import (
    dispatch_task,
    exercise_generation_task,
    ingest_url_task,
)

# ============================================================================
# VECTOR 1: Path Traversal & Workspace Manipulation
# ============================================================================


def test_poc_vuln_001_path_traversal_subfolder_creation(tmp_path: Path) -> None:
    """VULN-001: Path Traversal in folder creation via QDir.mkdir outside notes_dir is blocked."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    malicious_payload = "../../escaped_folder"
    # Verification: resolve_safe_path blocks relative path escape outside notes_dir
    with pytest.raises(ValueError, match="Path traversal detected"):
        resolve_safe_path(notes_dir, malicious_payload)

    # Verification: sanitize_filename strips traversal markers
    sanitized = sanitize_filename(malicious_payload)
    assert ".." not in sanitized
    assert "/" not in sanitized

    escaped_dir = tmp_path.parent / "escaped_folder"
    assert not escaped_dir.exists(), "Directory should not have escaped outside tmp_path"


def test_poc_vuln_002_arbitrary_file_read_overwrite(tmp_path: Path) -> None:
    """VULN-002: Arbitrary file load and blind overwrite outside notes workspace is blocked."""
    notes_dir = tmp_path / "notes"
    db_dir = tmp_path / "database"
    notes_dir.mkdir()
    db_dir.mkdir()

    sensitive_db = db_dir / "lang_attack.db"
    sensitive_db.write_text("ORIGINAL_SQLITE_HEADER", encoding="utf-8")

    # CenterDeck and LeftWorkspace validate target paths via resolve_safe_path
    target_path = notes_dir / "../database/lang_attack.db"
    with pytest.raises(ValueError, match="Path traversal detected"):
        resolve_safe_path(notes_dir, target_path)

    # Sensitive database remains untouched
    assert sensitive_db.read_text(encoding="utf-8") == "ORIGINAL_SQLITE_HEADER"


def test_poc_vuln_003_null_byte_path_injection(tmp_path: Path) -> None:
    """VULN-003: Embedded null byte (\x00) path injection is safely sanitized and rejected."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    null_byte_filename = "malicious\x00note.md"
    sanitized = sanitize_filename(null_byte_filename)
    assert "\x00" not in sanitized
    assert sanitized == "malicious_note.md"

    with pytest.raises(ValueError, match="Embedded null byte"):
        resolve_safe_path(notes_dir, null_byte_filename)


def test_poc_vuln_004_tmp_symlink_overwrite(tmp_path: Path) -> None:
    """VULN-004: Insecure fixed /tmp/ symlink overwrite is blocked by AudioRecorder and TTS."""
    from lang_attack.core.tts import synthesize_speech

    victim_file = tmp_path / "victim_data.txt"
    victim_file.write_text("CRITICAL_USER_DATA", encoding="utf-8")

    # AudioRecorder explicit audio_data symlink rejection
    symlink_path = tmp_path / "lang_attack_recorded.wav"
    os.symlink(victim_file, symlink_path)

    recorder = AudioRecorder(sample_rate=16000, channels=1)
    with pytest.raises(ValueError, match="Refusing to write audio to symbolic link"):
        recorder.save_wav(symlink_path, np.zeros(160, dtype=np.float32))

    # AudioRecorder default audio_data (calls stop_recording internally)
    with pytest.raises(ValueError, match="Refusing to write audio to symbolic link"):
        recorder.save_wav(symlink_path)

    # TTS speech synthesis symlink rejection
    symlink_tts = tmp_path / "tutor_reply.wav"
    os.symlink(victim_file, symlink_tts)
    with pytest.raises(ValueError, match="Refusing to write audio to symbolic link"):
        synthesize_speech("Hello tutor", symlink_tts)

    # The victim file was NOT overwritten
    assert victim_file.read_text(encoding="utf-8") == "CRITICAL_USER_DATA"


# ============================================================================
# VECTOR 2: Indirect Prompt Injection & Schema Validation
# ============================================================================


def test_poc_crash_001_youtube_ingest_attribute_error(tmp_path: Path) -> None:
    """CRASH-001: ingest_url_task handles list transcript without AttributeError."""
    with patch(
        "lang_attack.workers.tasks.extract_youtube_transcript",
        return_value=[{"text": "Sample text", "start": 0.0, "duration": 1.0}],
    ):
        out_result = ingest_url_task("https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)
        assert isinstance(out_result, dict)
        file_path = Path(out_result["path"])
        assert file_path.exists()
        assert "Sample text" in file_path.read_text(encoding="utf-8")


def test_poc_crash_002_exercise_task_signature_mismatch() -> None:
    """CRASH-002: exercise_generation_task succeeds with keyword arguments (no TypeError)."""
    with patch("lang_attack.workers.tasks.get_llm_provider") as mock_factory:
        mock_provider = mock_factory.return_value
        mock_provider.create_exercise_session.return_value = [
            ExerciseItem(
                id="ex-1",
                type=ExerciseType.CLOZE,
                prompt="The cat sat on the ___.",
                correct_answer="mat",
            )
        ]
        items = exercise_generation_task(
            topic="Grammar",
            note_content="The cat sat on the mat.",
            language="en",
            num_exercises=5,
        )
        assert len(items) == 1
        assert items[0].correct_answer == "mat"


def test_poc_crash_003_centerdeck_exercise_item_schema_mismatch() -> None:
    """CRASH-003: CenterDeck accesses question, context_sentence, acceptable_alternatives cleanly."""
    item = ExerciseItem(
        id="ex-1",
        type=ExerciseType.CLOZE,
        prompt="The quick brown ___ jumps over the lazy dog.",
        correct_answer="fox",
        context_sentence="The quick brown fox jumps over the lazy dog.",
    )

    assert item.question == "The quick brown ___ jumps over the lazy dog."
    assert item.context_sentence == "The quick brown fox jumps over the lazy dog."
    assert item.acceptable_alternatives == []


# ============================================================================
# VECTOR 3: Audio Fuzzing & Memory Corruption
# ============================================================================


def test_poc_crash_004_whisper_0byte_wav_uncaught_eof_error(tmp_path: Path) -> None:
    """CRASH-004: WhisperASR.transcribe safely raises ValueError on 0-byte WAV files."""
    empty_wav = tmp_path / "zero_bytes.wav"
    empty_wav.touch()

    asr = WhisperASR()
    with pytest.raises(ValueError, match="Corrupted or empty audio file"):
        asr.transcribe(empty_wav)


def test_poc_vuln_005_120db_static_noise_hallucination(tmp_path: Path) -> None:
    """VULN-005: 120dB static noise clipping is identified and rejected with score < 20.0."""
    noise_wav = tmp_path / "noise_120db.wav"
    rec = AudioRecorder(sample_rate=16000, channels=1)

    # 120dB clipping noise: random float32 signal exceeding [-1.0, 1.0]
    high_noise = np.random.uniform(-10.0, 10.0, 16000 * 2).astype(np.float32)
    rec.save_wav(noise_wav, high_noise)

    result = evaluate_speech("The quick brown fox jumps over the lazy dog.", noise_wav)
    assert result.overall_score < 20.0, f"Expected low score for noise, got {result.overall_score}"
    assert result.fluency_score is not None and result.fluency_score < 20.0
    assert result.recognized_text == "pure noise detected"


def test_poc_crash_005_audio_nan_inf_pydantic_validation_failure() -> None:
    """CRASH-005: Audio buffers with NaN or Inf are caught by Pydantic with descriptive validation error."""
    with pytest.raises(pydantic_core.ValidationError):
        PhonemeEvaluation(phoneme="a", score=float("nan"), status="correct")

    with pytest.raises(pydantic_core.ValidationError):
        PhonemeEvaluation(phoneme="a", score=float("inf"), status="correct")

    valid_eval = PhonemeEvaluation(phoneme="a", score=90.0, status="correct")
    assert valid_eval.score == 90.0


# ============================================================================
# VECTOR 4: Concurrency Stress & Resource Exhaustion
# ============================================================================


def test_poc_crash_006_rightlab_pedagogical_feedback_mismatch() -> None:
    """CRASH-006: RightLab accesses pedagogical_feedback on SpeechAssessmentResult safely."""
    result = SpeechAssessmentResult(
        overall_score=85.0,
        feedback_tips=["Speak more slowly and clearly."],
    )
    assert hasattr(result, "pedagogical_feedback")
    assert result.pedagogical_feedback == "Speak more slowly and clearly."


def test_poc_vuln_006_dispatch_task_duplicate_signal_amplification(qapp: Any) -> None:
    """VULN-006: dispatch_task connects on_result exclusively to result signal, firing exactly once."""
    call_count = 0

    def test_callback(res: Any) -> None:
        nonlocal call_count
        call_count += 1

    def noop_task() -> str:
        return "SUCCESS"

    dispatch_task(noop_task, on_result=test_callback)

    QThreadPool.globalInstance().waitForDone(3000)
    qapp.processEvents()

    assert call_count == 1, f"Expected exactly 1 callback invocation, got {call_count}"


def test_poc_crash_007_phonetic_repository_column_mismatch() -> None:
    """CRASH-007: PhoneticRepository.list_history executes without column mismatch error."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    repo = PhoneticRepository(conn)

    results = repo.list_history(min_gop=50.0)
    assert isinstance(results, list)
    assert len(results) == 0


def test_poc_vuln_007_massive_string_ui_rendering_exhaustion(qapp: Any) -> None:
    """VULN-007: Massive phoneme list is capped to prevent QTextBrowser rendering exhaustion."""
    lab = RightLab()
    phonemes = [{"symbol": "a", "score": 80.0}] * 10000  # 10,000 phonemes

    html = lab.render_phonetic_html(phonemes)
    assert html.count("<span") <= 250, f"Expected <= 250 spans due to pagination guard, got {html.count('<span')}"
    lab.phonetic_display.setHtml(html)
    assert len(lab.phonetic_display.toPlainText()) < 5000


def test_poc_crash_008_audio_recorder_deadlock_on_stop() -> None:
    """CRASH-008: AudioRecorder.stop_recording() does not deadlock due to re-entrant locking."""
    import threading

    recorder = AudioRecorder(sample_rate=16000, channels=1)
    recorder.start_recording()

    thread = threading.Thread(target=recorder.stop_recording, daemon=True)
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "AudioRecorder.stop_recording() deadlocked on self._lock re-entry"


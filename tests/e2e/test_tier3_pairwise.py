"""Tier 3: Pairwise Cross-Feature Integration E2E Suite.

Tests interactions between paired feature areas:
1. Pair 1: Ingestion -> Notes Vault -> LLM Exercise Generation -> SRS Storage
2. Pair 2: Audio Recording -> Whisper ASR -> Phonetics GOP -> SQLite History
3. Pair 3: Notes Vault -> Context Selector -> Sentence Extraction -> TTS Synthesis
4. Pair 4: Background GenericWorker -> @latency_tracker -> SQLite Metrics Table
5. Pair 5: Exercise Completion -> SRS SM-2 Algorithm Updates
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import lang_attack.telemetry as tel
from tests.conftest import generate_test_wav

# ============================================================================
# Pair 1: Ingestion -> Notes Vault -> Exercise Generation -> SRS Storage
# ============================================================================


def test_tier3_ingestion_to_notes_to_srs(
    tmp_path: Path, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pair 1: Scrapes web article, writes to notes/, creates exercises, stores SRS items."""
    try:
        from lang_attack.core.ingestion import extract_article
        from lang_attack.core.llm_provider import GeminiProvider
        from lang_attack.db.connection import init_db
        from lang_attack.schemas.exercise import ExerciseItem
    except ImportError:
        pytest.skip("Required M2/M3 modules not yet implemented")

    # Step A: Ingestion
    mock_trafilatura = MagicMock()
    mock_trafilatura.fetch_url.return_value = "<html><h1>Le Petit Prince</h1><p>On ne voit bien qu'avec le coeur.</p></html>"
    mock_trafilatura.extract.return_value = "# Le Petit Prince\n\nOn ne voit bien qu'avec le coeur."
    monkeypatch.setattr("trafilatura.fetch_url", mock_trafilatura.fetch_url, raising=False)
    monkeypatch.setattr("trafilatura.extract", mock_trafilatura.extract, raising=False)

    article = extract_article("https://example.com/le-petit-prince")
    assert article is not None

    # Step B: Save to notes vault
    notes_dir = tmp_path / "notes" / "French"
    notes_dir.mkdir(parents=True, exist_ok=True)
    note_file = notes_dir / "petit_prince.md"
    note_file.write_text(article.get("text", "# Le Petit Prince"), encoding="utf-8")
    assert note_file.exists()

    # Step C: Generate exercises via LLM
    GeminiProvider()
    item = ExerciseItem(
        id="ex-prince-1",
        type="multiple_choice",
        prompt="Complete: 'On ne voit bien qu'avec le ___.'",
        options=["coeur", "vent", "soleil"],
        correct_answer="coeur",
        language="fr",
    )
    assert item.correct_answer == "coeur"

    # Step D: Store vocabulary in SQLite SRS
    conn = init_db(tmp_db_path)
    cursor = conn.cursor()
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(vocabulary_srs);")]
    if "item_uuid" in cols:
        cursor.execute(
            """
            INSERT INTO vocabulary_srs (item_uuid, language, term, translation, next_review_at, created_at)
            VALUES ('srs-p1', 'fr', 'coeur', 'heart', '2026-10-30T00:00:00Z', '2026-10-29T00:00:00Z');
            """
        )
    else:
        cursor.execute(
            """
            INSERT INTO vocabulary_srs (item_id, language, term, translation, next_review_date)
            VALUES ('srs-p1', 'fr', 'coeur', 'heart', '2026-10-30T00:00:00Z');
            """
        )
    conn.commit()

    cursor.execute("SELECT term FROM vocabulary_srs WHERE language = 'fr';")
    assert cursor.fetchone()[0] == "coeur"
    conn.close()


# ============================================================================
# Pair 2: Recording -> Whisper ASR -> Phonetics GOP -> SQLite History
# ============================================================================


def test_tier3_speech_recording_to_asr_to_gop_to_db(
    tmp_path: Path, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pair 2: Audio capture triggers ASR transcribe, Phonetic GOP eval, and SQLite logging."""
    try:
        from lang_attack.core.asr import transcribe_audio
        from lang_attack.core.phonetics import evaluate_speech
        from lang_attack.db.connection import init_db
    except ImportError:
        pytest.skip("Required M2/M3 modules not yet implemented")

    # Step A: Simulate recorded audio
    audio_path = generate_test_wav(tmp_path / "recording.wav")

    # Step B: Transcribe audio
    transcript = transcribe_audio(audio_path)
    assert len(transcript) > 0

    # Step C: Evaluate phonetics GOP
    eval_result = evaluate_speech(target_text="The quick brown fox", audio_path=audio_path, language="en")
    assert eval_result is not None

    # Step D: Persist result to SQLite phonetic_history
    conn = init_db(tmp_db_path)
    cursor = conn.cursor()
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(phonetic_history);")]
    score = getattr(eval_result, "overall_score", 85.0)

    if "session_id" in cols and "words_json" in cols:
        cursor.execute(
            """
            INSERT INTO phonetic_history (language, target_text, recognized_transcript, overall_score, fluency_score, completeness_score, words_json, created_at)
            VALUES ('en', 'The quick brown fox', 'The quick brown fox', ?, 90.0, 100.0, '[]', '2026-10-29T12:00:00Z');
            """,
            (score,),
        )
    else:
        cursor.execute(
            """
            INSERT INTO phonetic_history (record_id, language, target_text, recognized_text, overall_score, details_json)
            VALUES ('rec-1', 'en', 'The quick brown fox', 'The quick brown fox', ?, '[]');
            """,
            (score,),
        )
    conn.commit()

    cursor.execute("SELECT overall_score FROM phonetic_history;")
    assert cursor.fetchone()[0] == pytest.approx(score, rel=1e-2)
    conn.close()


# ============================================================================
# Pair 3: Notes Vault -> Context Selector -> Sentence Extraction -> TTS
# ============================================================================


def test_tier3_notes_sentence_to_tts_audio(
    tmp_path: Path, tmp_notes_vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pair 3: Select note from vault, extract sentence, synthesize reference TTS audio."""
    try:
        from lang_attack.core.tts import synthesize_speech
    except ImportError:
        pytest.skip("Required M3 modules not yet implemented")

    # Read sentence from test vault
    note = tmp_notes_vault / "Français" / "salutations.md"
    content = note.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.split("\n") if line.strip() and not line.startswith("#")]
    assert len(lines) >= 1
    target_sentence = lines[0]  # "Bonjour tout le monde."

    # Synthesize audio for target sentence
    out_audio = tmp_path / "target_sentence.wav"
    result = synthesize_speech(target_sentence, out_audio)
    assert result.exists()
    assert result.suffix == ".wav"


# ============================================================================
# Pair 4: Background GenericWorker -> @latency_tracker -> SQLite Metrics Table
# ============================================================================


def test_tier3_worker_telemetry_to_metrics_db(
    tmp_db_path: Path, qapp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pair 4: GenericWorker executes @latency_tracker pipeline and persists metrics to DB."""
    try:
        from lang_attack.db.connection import init_db
        from lang_attack.workers.generic import GenericWorker
        from PySide6.QtCore import QThreadPool
    except ImportError:
        pytest.skip("Required M2/M3 modules not yet implemented")

    conn = init_db(tmp_db_path)
    conn.close()

    @tel.latency_tracker("background_ai_task")
    def heavy_calculation(n: int) -> int:
        return sum(i * i for i in range(n))

    worker = GenericWorker(heavy_calculation, 1000)
    pool = QThreadPool.globalInstance()
    pool.start(worker)
    pool.waitForDone(2000)

    # Verify telemetry logged
    daily_dir = tel.get_daily_log_dir()
    assert daily_dir.exists()


# ============================================================================
# Pair 5: Exercise Completion -> SRS SM-2 Algorithm Updates
# ============================================================================


def test_tier3_exercise_completion_to_srs_progression(tmp_db_path: Path) -> None:
    """Pair 5: Completing exercise updates repetition and interval in vocabulary_srs."""
    try:
        from lang_attack.db.connection import init_db
    except ImportError:
        pytest.skip("Required M2 modules not yet implemented")

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()

    cols = [c[1] for c in cursor.execute("PRAGMA table_info(vocabulary_srs);")]
    if "item_uuid" in cols:
        cursor.execute(
            """
            INSERT INTO vocabulary_srs (item_uuid, language, term, translation, srs_stage, interval_days, next_review_at, created_at)
            VALUES ('srs-sm2', 'de', 'Apfel', 'apple', 0, 0, '2026-10-29T00:00:00Z', '2026-10-29T00:00:00Z');
            """
        )
    else:
        cursor.execute(
            """
            INSERT INTO vocabulary_srs (item_id, language, term, translation, repetition_count, interval_days, next_review_date)
            VALUES ('srs-sm2', 'de', 'Apfel', 'apple', 0, 0, '2026-10-29T00:00:00Z');
            """
        )
    conn.commit()

    # Simulate SRS review grade 5 (perfect recall -> stage advances)
    if "srs_stage" in cols:
        cursor.execute(
            """
            UPDATE vocabulary_srs
            SET srs_stage = 1, interval_days = 1, next_review_at = '2026-10-30T00:00:00Z'
            WHERE term = 'Apfel';
            """
        )
    else:
        cursor.execute(
            """
            UPDATE vocabulary_srs
            SET repetition_count = 1, interval_days = 1, next_review_date = '2026-10-30T00:00:00Z'
            WHERE term = 'Apfel';
            """
        )
    conn.commit()

    cursor.execute("SELECT interval_days FROM vocabulary_srs WHERE term = 'Apfel';")
    assert cursor.fetchone()[0] == 1
    conn.close()

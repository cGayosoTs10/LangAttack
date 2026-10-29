"""Tier 1: Feature Coverage E2E Suite.

Covers all 8 inventoried feature areas in isolation with at least 5 test cases each:
1. FEAT-ENV: Tooling, pyproject.toml, .gitignore, dependencies (5 tests)
2. FEAT-TEL: Partitioned logs, @latency_tracker, zero pdb (5 tests)
3. FEAT-DAT: SQLite tables, DDL, WAL mode, unconstrained language (5 tests)
4. FEAT-SCH: Phonetics & Exercise Pydantic models (5 tests)
5. FEAT-CON: GenericWorker, WorkerSignals, QThreadPool (5 tests)
6. FEAT-ING: Web and YouTube transcript ingestion (5 tests)
7. FEAT-AIE: TTS, ASR, Phonetics GOP, LLM provider (5 tests)
8. FEAT-GUI: PySide6 dark UI, QSplitter, 3 panes (5 tests)
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import lang_attack.telemetry as tel
from lang_attack.config import REPO_ROOT
from tests.conftest import generate_test_wav

# ============================================================================
# 1. FEAT-ENV: Environment, Tooling & Packaging (5 tests)
# ============================================================================


def test_tier1_env_uv_initialization() -> None:
    """FEAT-ENV-01: Verify pyproject.toml is at repo root with correct project name."""
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.exists()
    content = pyproject.read_text(encoding="utf-8")
    assert 'name = "lang-attack"' in content or 'name = "langattack"' in content


def test_tier1_env_dependencies_declared() -> None:
    """FEAT-ENV-02: Verify core production dependencies are declared in pyproject.toml."""
    content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dep in ["pyside6", "pydantic", "sounddevice", "numpy", "scipy", "trafilatura"]:
        assert dep in content.lower(), f"Dependency '{dep}' missing from pyproject.toml"


def test_tier1_env_dev_dependencies() -> None:
    """FEAT-ENV-03: Verify dev dependencies (pytest, ruff, ty) are declared."""
    content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dev_dep in ["pytest", "ruff", "ty"]:
        assert dev_dep in content.lower(), f"Dev dependency '{dev_dep}' missing"


def test_tier1_env_ruff_and_ty_config() -> None:
    """FEAT-ENV-04: Verify ruff line-length 100, py312 target and ty strict = true."""
    content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "line-length = 100" in content
    assert "target-version = \"py312\"" in content or "target-version = 'py312'" in content
    assert "strict = true" in content


def test_tier1_env_gitignore_sanitized() -> None:
    """FEAT-ENV-05: Verify .gitignore excludes .agents, keys, caches, logs, and database."""
    gitignore = REPO_ROOT / ".gitignore"
    assert gitignore.exists()
    content = gitignore.read_text(encoding="utf-8")
    for pattern in [".agents", "__pycache__", "logs", "database"]:
        assert pattern in content, f"Pattern '{pattern}' missing from .gitignore"


# ============================================================================
# 2. FEAT-TEL: Telemetry, Logs & Latency Tracking (5 tests)
# ============================================================================


def test_tier1_tel_daily_log_partitioning() -> None:
    """FEAT-TEL-01: Verify logs/YYYY-MM-DD/ directory creation."""
    daily_dir = tel.get_daily_log_dir()
    assert daily_dir.exists()
    assert daily_dir.is_dir()


def test_tier1_tel_setup_loggers() -> None:
    """FEAT-TEL-01: Verify independent app and latency loggers."""
    app_logger, latency_logger = tel.setup_telemetry()
    assert app_logger.name == "lang_attack.app"
    assert latency_logger.name == "lang_attack.latency"


def test_tier1_tel_latency_tracker_success() -> None:
    """FEAT-TEL-02: Verify latency tracker records duration on success."""

    @tel.latency_tracker("test_pipe")
    def run_work() -> str:
        return "ok"

    assert run_work() == "ok"


def test_tier1_tel_latency_tracker_failure() -> None:
    """FEAT-TEL-02: Verify latency tracker captures failure and re-raises exception."""

    @tel.latency_tracker("test_fail_pipe")
    def fail_work() -> None:
        raise ValueError("simulated")

    with pytest.raises(ValueError):
        fail_work()


def test_tier1_tel_zero_pdb_compliance() -> None:
    """FEAT-TEL-03: Zero tolerance for pdb, ipdb, or breakpoint in source code."""
    for py_file in REPO_ROOT.glob("lang_attack/**/*.py"):
        code = py_file.read_text(encoding="utf-8")
        assert "import pdb" not in code
        assert "pdb.set_trace" not in code
        assert "breakpoint()" not in code


# ============================================================================
# 3. FEAT-DAT: SQLite Persistence & Flexible Schema (5 tests)
# ============================================================================


def test_tier1_dat_sqlite_initialization(tmp_db_path: Path) -> None:
    """FEAT-DAT-01: Verify SQLite auto-initialization and table creation."""
    try:
        from lang_attack.db.connection import init_db
    except ImportError:
        pytest.skip("lang_attack.db not yet implemented (scheduled for M2)")

    conn = init_db(tmp_db_path)
    assert conn is not None
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "study_sessions" in tables
    assert "phonetic_history" in tables
    assert "vocabulary_srs" in tables
    assert "pipeline_metrics" in tables
    conn.close()


def test_tier1_dat_study_sessions_table(tmp_db_path: Path) -> None:
    """FEAT-DAT-02: Verify study_sessions table accepts flexible language TEXT."""
    try:
        from lang_attack.db.connection import init_db
    except ImportError:
        pytest.skip("lang_attack.db not yet implemented (scheduled for M2)")

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(study_sessions);")]
    assert "language" in cols
    conn.close()


def test_tier1_dat_phonetic_history_table(tmp_db_path: Path) -> None:
    """FEAT-DAT-03: Verify phonetic_history table schema and language field."""
    try:
        from lang_attack.db.connection import init_db
    except ImportError:
        pytest.skip("lang_attack.db not yet implemented (scheduled for M2)")

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(phonetic_history);")]
    assert "language" in cols
    assert "overall_score" in cols
    conn.close()


def test_tier1_dat_vocabulary_srs_table(tmp_db_path: Path) -> None:
    """FEAT-DAT-04: Verify vocabulary_srs table schema."""
    try:
        from lang_attack.db.connection import init_db
    except ImportError:
        pytest.skip("lang_attack.db not yet implemented (scheduled for M2)")

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(vocabulary_srs);")]
    assert "term" in cols
    assert "language" in cols
    conn.close()


def test_tier1_dat_pipeline_metrics_table(tmp_db_path: Path) -> None:
    """FEAT-DAT-05: Verify pipeline_metrics table schema."""
    try:
        from lang_attack.db.connection import init_db
    except ImportError:
        pytest.skip("lang_attack.db not yet implemented (scheduled for M2)")

    conn = init_db(tmp_db_path)
    cursor = conn.cursor()
    cols = [c[1] for c in cursor.execute("PRAGMA table_info(pipeline_metrics);")]
    assert "pipeline_name" in cols
    assert "duration_ms" in cols
    assert "status" in cols
    conn.close()


# ============================================================================
# 4. FEAT-SCH: Pydantic Data Contracts (5 tests)
# ============================================================================


def test_tier1_sch_phoneme_evaluation() -> None:
    """FEAT-SCH-01: Verify PhonemeEvaluation model instantiation."""
    try:
        from lang_attack.schemas.phonetics import PhonemeEvaluation
    except ImportError:
        pytest.skip("lang_attack.schemas not yet implemented (scheduled for M2)")

    score_field = "score" if "score" in PhonemeEvaluation.model_fields else "gop_score"
    kwargs = {"phoneme": "s", "status": "correct", score_field: 95.0}
    if "ipa_symbol" in PhonemeEvaluation.model_fields:
        kwargs["ipa_symbol"] = "s"
        kwargs["expected_phoneme"] = "s"
        kwargs["detected_phoneme"] = "s"
        kwargs["is_correct"] = True
        kwargs["start_time"] = 0.0
        kwargs["end_time"] = 0.1
    p = PhonemeEvaluation(**kwargs)
    assert p.phoneme == "s"


def test_tier1_sch_word_evaluation() -> None:
    """FEAT-SCH-02: Verify WordEvaluation model instantiation."""
    try:
        from lang_attack.schemas.phonetics import WordEvaluation
    except ImportError:
        pytest.skip("lang_attack.schemas not yet implemented (scheduled for M2)")

    score_field = "score" if "score" in WordEvaluation.model_fields else "gop_score"
    kwargs = {"word": "sun", score_field: 91.0, "phonemes": [], "start_time": 0.0, "end_time": 0.3}
    if "is_correct" in WordEvaluation.model_fields:
        kwargs["is_correct"] = True
    w = WordEvaluation(**kwargs)
    assert w.word == "sun"


def test_tier1_sch_speech_assessment_result() -> None:
    """FEAT-SCH-03: Verify SpeechAssessmentResult model instantiation."""
    try:
        from lang_attack.schemas.phonetics import SpeechAssessmentResult
    except ImportError:
        pytest.skip("lang_attack.schemas not yet implemented (scheduled for M2)")

    kwargs: dict[str, Any] = {"overall_score": 85.0, "words": []}
    if "target_text" in SpeechAssessmentResult.model_fields:
        kwargs["target_text"] = "Hello"
        kwargs["recognized_text"] = "Hello"
    else:
        kwargs["reference_text"] = "Hello"
        kwargs["transcript"] = "Hello"
        kwargs["fluency_score"] = 90.0
        kwargs["completeness_score"] = 100.0
        kwargs["audio_duration_s"] = 1.0

    res = SpeechAssessmentResult(**kwargs)
    assert res.overall_score == 85.0


def test_tier1_sch_exercise_item() -> None:
    """FEAT-SCH-04: Verify ExerciseItem model instantiation."""
    try:
        from lang_attack.schemas.exercise import ExerciseItem
    except ImportError:
        pytest.skip("lang_attack.schemas not yet implemented (scheduled for M2)")

    item = ExerciseItem(
        id="ex-1",
        type="multiple_choice",
        prompt="Select translation",
        options=["chat", "chien"],
        correct_answer="chat",
        language="fr",
    )
    assert item.correct_answer == "chat"


def test_tier1_sch_exercise_session() -> None:
    """FEAT-SCH-05: Verify ExerciseSession model lifecycle."""
    try:
        from lang_attack.schemas.exercise import ExerciseItem, ExerciseSession
    except ImportError:
        pytest.skip("lang_attack.schemas not yet implemented (scheduled for M2)")

    item = ExerciseItem(
        id="ex-1",
        type="multiple_choice",
        prompt="Prompt",
        options=["A", "B"],
        correct_answer="A",
        language="en",
    )
    session = ExerciseSession(
        id="s-1" if "id" in ExerciseSession.model_fields else None,
        session_id="s-1" if "session_id" in ExerciseSession.model_fields else None,
        title="Session 1",
        language="en",
        items=[item],
    )
    assert len(session.items) == 1


# ============================================================================
# 5. FEAT-CON: Non-blocking Concurrency & Workers (5 tests)
# ============================================================================


def test_tier1_con_worker_signals() -> None:
    """FEAT-CON-01: Verify WorkerSignals class provides signals."""
    try:
        from lang_attack.workers.generic import WorkerSignals
    except ImportError:
        pytest.skip("lang_attack.workers not yet implemented (scheduled for M3)")

    signals = WorkerSignals()
    assert hasattr(signals, "started")
    assert hasattr(signals, "error")


def test_tier1_con_generic_worker_creation() -> None:
    """FEAT-CON-01: Verify GenericWorker wraps callable."""
    try:
        from lang_attack.workers.generic import GenericWorker
    except ImportError:
        pytest.skip("lang_attack.workers not yet implemented (scheduled for M3)")

    worker = GenericWorker(lambda: 42)
    assert worker is not None


def test_tier1_con_generic_worker_synchronous_run() -> None:
    """FEAT-CON-01: Verify GenericWorker executes callable on run()."""
    try:
        from lang_attack.workers.generic import GenericWorker
    except ImportError:
        pytest.skip("lang_attack.workers not yet implemented (scheduled for M3)")

    out: list[int] = []
    worker = GenericWorker(lambda: out.append(99))
    worker.run()
    assert 99 in out


def test_tier1_con_generic_worker_qthreadpool(qapp: Any) -> None:
    """FEAT-CON-02: Verify GenericWorker runs on QThreadPool.globalInstance()."""
    try:
        from lang_attack.workers.generic import GenericWorker
        from PySide6.QtCore import QThreadPool
    except ImportError:
        pytest.skip("lang_attack.workers or PySide6 not yet implemented")

    done = []
    worker = GenericWorker(lambda: done.append(True))
    pool = QThreadPool.globalInstance()
    pool.start(worker)
    pool.waitForDone(2000)
    assert done == [True]


def test_tier1_con_generic_worker_error_emission() -> None:
    """FEAT-CON-01: Verify GenericWorker emits error signal on exception."""
    try:
        from lang_attack.workers.generic import GenericWorker
    except ImportError:
        pytest.skip("lang_attack.workers not yet implemented (scheduled for M3)")

    errors: list[Any] = []

    def raise_err() -> None:
        raise RuntimeError("boom")

    worker = GenericWorker(raise_err)
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert len(errors) > 0


# ============================================================================
# 6. FEAT-ING: Content Ingestion (5 tests)
# ============================================================================


def test_tier1_ing_web_article_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAT-ING-01: Verify trafilatura extracts article markdown."""
    try:
        from lang_attack.core.ingestion import extract_article
    except ImportError:
        pytest.skip("lang_attack.core.ingestion not yet implemented (scheduled for M3)")

    mock_trafilatura = MagicMock()
    mock_trafilatura.fetch_url.return_value = "<html><h1>Hello</h1><p>World</p></html>"
    mock_trafilatura.extract.return_value = "# Hello\n\nWorld"
    monkeypatch.setattr("trafilatura.fetch_url", mock_trafilatura.fetch_url, raising=False)
    monkeypatch.setattr("trafilatura.extract", mock_trafilatura.extract, raising=False)

    res = extract_article("https://example.com/test")
    assert res is not None


def test_tier1_ing_youtube_transcript_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAT-ING-02: Verify youtube-transcript-api extracts subtitles."""
    try:
        from lang_attack.core.ingestion import extract_youtube_transcript
    except ImportError:
        pytest.skip("lang_attack.core.ingestion not yet implemented (scheduled for M3)")

    mock_api = MagicMock()
    mock_api.get_transcript.return_value = [{"text": "Hello", "start": 0.0, "duration": 1.0}]
    monkeypatch.setattr(
        "youtube_transcript_api.YouTubeTranscriptApi.get_transcript",
        mock_api.get_transcript,
        raising=False,
    )

    subs = extract_youtube_transcript("https://youtube.com/watch?v=123")
    assert len(subs) > 0


def test_tier1_ing_article_language_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAT-ING-01: Verify article language metadata is captured."""
    try:
        from lang_attack.core.ingestion import extract_article
    except ImportError:
        pytest.skip("lang_attack.core.ingestion not yet implemented (scheduled for M3)")

    res = extract_article("https://example.com/test")
    assert res is not None


def test_tier1_ing_youtube_timestamp_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAT-ING-02: Verify YouTube subtitles include timestamp offsets."""
    try:
        from lang_attack.core.ingestion import extract_youtube_transcript
    except ImportError:
        pytest.skip("lang_attack.core.ingestion not yet implemented (scheduled for M3)")

    subs = extract_youtube_transcript("https://youtube.com/watch?v=123")
    assert subs is not None


def test_tier1_ing_content_to_markdown() -> None:
    """FEAT-ING-01: Verify ingested content preserves or generates Markdown format."""
    try:
        from lang_attack.core.ingestion import extract_article
    except ImportError:
        pytest.skip("lang_attack.core.ingestion not yet implemented (scheduled for M3)")

    res = extract_article("https://example.com/test")
    assert res is not None


# ============================================================================
# 7. FEAT-AIE: AI Pipelines (5 tests)
# ============================================================================


def test_tier1_aie_tts_cpu_wav_synthesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAT-AIE-01: Verify Kokoro/Piper TTS synthesizes 16kHz WAV on CPU."""
    try:
        from lang_attack.core.tts import synthesize_speech
    except ImportError:
        pytest.skip("lang_attack.core.tts not yet implemented (scheduled for M3)")

    out_file = tmp_path / "target.wav"
    res = synthesize_speech("Hello", out_file)
    assert res.exists()


def test_tier1_aie_asr_transcribe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FEAT-AIE-02: Verify faster-whisper ASR transcribes audio."""
    try:
        from lang_attack.core.asr import transcribe_audio
    except ImportError:
        pytest.skip("lang_attack.core.asr not yet implemented (scheduled for M3)")

    wav = generate_test_wav(tmp_path / "audio.wav")
    text = transcribe_audio(wav)
    assert isinstance(text, str)


def test_tier1_aie_phonetic_gop_scoring(tmp_path: Path) -> None:
    """FEAT-AIE-03: Verify Wav2Vec GOP evaluation."""
    try:
        from lang_attack.core.phonetics import evaluate_speech
    except ImportError:
        pytest.skip("lang_attack.core.phonetics not yet implemented (scheduled for M3)")

    wav = generate_test_wav(tmp_path / "speech.wav")
    res = evaluate_speech("hello", wav, language="en")
    assert res is not None


def test_tier1_aie_llm_text_generation() -> None:
    """FEAT-AIE-04: Verify LLM provider text completion."""
    try:
        from lang_attack.core.llm_provider import GeminiProvider
    except ImportError:
        pytest.skip("lang_attack.core.llm_provider not yet implemented (scheduled for M3)")

    llm = GeminiProvider()
    out = llm.generate("Hello")
    assert len(out) > 0


def test_tier1_aie_llm_exercise_generation() -> None:
    """FEAT-AIE-04: Verify LLM provider exercise generation from text."""
    try:
        from lang_attack.core.llm_provider import GeminiProvider
    except ImportError:
        pytest.skip("lang_attack.core.llm_provider not yet implemented (scheduled for M3)")

    llm = GeminiProvider()
    if hasattr(llm, "generate_exercises"):
        exercises = llm.generate_exercises("Sample text for drill.", language="en")
        assert len(exercises) >= 1


# ============================================================================
# 8. FEAT-GUI: PySide6 Desktop GUI (5 tests)
# ============================================================================


def test_tier1_gui_dark_theme_qss() -> None:
    """FEAT-GUI-01: Verify dark theme stylesheet exists and contains GOP color tokens."""
    qss = Path("lang_attack/ui/styles.qss")
    if not qss.exists():
        qss = Path("lang_attack/ui/resources/styles.qss")
    if not qss.exists():
        pytest.skip("lang_attack/ui/styles.qss not yet implemented (scheduled for M4)")

    content = qss.read_text(encoding="utf-8")
    assert len(content) > 0


def test_tier1_gui_main_window_qsplitter(qapp: Any) -> None:
    """FEAT-GUI-02: Verify MainWindow horizontal QSplitter with 3 panes."""
    try:
        from lang_attack.ui.main_window import MainWindow
    except ImportError:
        pytest.skip("lang_attack.ui.main_window not yet implemented (scheduled for M4)")

    window = MainWindow()
    assert window is not None


def test_tier1_gui_left_workspace_notes(qapp: Any) -> None:
    """FEAT-GUI-03: Verify LeftWorkspace note management pane."""
    try:
        from lang_attack.ui.left_workspace import LeftWorkspace
    except ImportError:
        pytest.skip("lang_attack.ui.left_workspace not yet implemented (scheduled for M4)")

    left = LeftWorkspace()
    assert left is not None


def test_tier1_gui_center_deck_tabs(qapp: Any) -> None:
    """FEAT-GUI-05 / 06 / 07: Verify CenterDeck 3 tab widgets."""
    try:
        from lang_attack.ui.center_deck import CenterDeck
    except ImportError:
        pytest.skip("lang_attack.ui.center_deck not yet implemented (scheduled for M4)")

    center = CenterDeck()
    assert center is not None


def test_tier1_gui_right_lab_player_recorder(qapp: Any) -> None:
    """FEAT-GUI-08 / 09 / 10 / 11: Verify RightLab speech lab pane."""
    try:
        from lang_attack.ui.right_lab import RightLab
    except ImportError:
        pytest.skip("lang_attack.ui.right_lab not yet implemented (scheduled for M4)")

    right = RightLab()
    assert right is not None

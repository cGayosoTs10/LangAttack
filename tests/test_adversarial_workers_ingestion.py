"""Empirical Challenger Milestone 3 Stress & Ingestion Verification Test Suite.

Adversarially challenges:
1. GenericWorker in QThreadPool: 20+ simultaneous workers, high-load burst concurrency,
   cancellation lifecycle (pre-execution and mid-execution), signal integrity (started,
   progress, finished, result, error), signature inspection safety.
2. extract_article: invalid URL schemes, malformed inputs, 404/500/timeout handling,
   empty and degenerate bodies, non-HTML responses, metadata fallback heuristics.
3. extract_youtube_transcript: video ID extraction patterns, invalid IDs, missing subtitles,
   API exceptions, malformed timestamp parsing, HTML entity unescaping.
4. Concurrent ingestion in worker pool: 20 simultaneous ingestion workers running on
   QThreadPool verifying non-blocking execution and thread safety.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Ensure system PySide6 location is discoverable in offscreen headless test environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"
for candidate in [
    f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
    str(
        Path.home()
        / f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    ),
    "/usr/lib/python3.14/site-packages",
]:
    if os.path.exists(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from PySide6.QtCore import QCoreApplication, QThreadPool  # noqa: E402

from lang_attack.core.ingestion import (  # noqa: E402
    chunk_article_text,
    extract_article,
    extract_video_id,
    extract_youtube_transcript,
    transcript_to_markdown,
    transcript_to_text,
)
from lang_attack.workers.generic import GenericWorker  # noqa: E402

# ============================================================================
# 1. GenericWorker & QThreadPool Concurrency Stress Tests
# ============================================================================


def test_generic_worker_20_simultaneous_qthreadpool(qapp: Any) -> None:
    """Stress test: 20 simultaneous workers running in QThreadPool.

    Dispatches a heterogeneous mix:
    - 5 compute-intensive workers
    - 5 sleep/I/O workers
    - 5 progress-reporting workers
    - 5 error-raising workers
    Verifies all signals (started, progress, finished, result, error) fire without deadlocks.
    """
    pool = QThreadPool.globalInstance()
    workers: list[GenericWorker] = []

    stats = {
        "started": 0,
        "finished": 0,
        "result": 0,
        "error": 0,
        "progress_count": 0,
    }
    lock = threading.Lock()

    def on_started() -> None:
        with lock:
            stats["started"] += 1

    def on_finished(res: Any) -> None:
        with lock:
            stats["finished"] += 1

    def on_result(res: Any) -> None:
        with lock:
            stats["result"] += 1

    def on_error(err_type: str, err_msg: str) -> None:
        with lock:
            stats["error"] += 1

    def on_progress(pct: int, msg: str) -> None:
        with lock:
            stats["progress_count"] += 1

    # Workloads
    def compute_fn(n: int) -> int:
        return sum(i * i for i in range(n))

    def sleep_fn(duration: float) -> str:
        time.sleep(duration)
        return "slept"

    def progress_fn(steps: int, progress_callback: Any = None) -> str:
        for i in range(1, steps + 1):
            if progress_callback:
                progress_callback(int((i / steps) * 100), f"step {i}")
            time.sleep(0.002)
        return "done"

    def error_fn(idx: int) -> None:
        time.sleep(0.002)
        raise ValueError(f"Deliberate error in worker {idx}")

    # Build 20 workers
    for i in range(5):
        w = GenericWorker(compute_fn, 500 + i)
        w.signals.started.connect(on_started)
        w.signals.finished.connect(on_finished)
        w.signals.result.connect(on_result)
        w.signals.error.connect(on_error)
        workers.append(w)

    for _ in range(5):
        w = GenericWorker(sleep_fn, 0.01)
        w.signals.started.connect(on_started)
        w.signals.finished.connect(on_finished)
        w.signals.result.connect(on_result)
        w.signals.error.connect(on_error)
        workers.append(w)

    for _ in range(5):
        w = GenericWorker(progress_fn, 4, progress_enabled=True)
        w.signals.started.connect(on_started)
        w.signals.progress.connect(on_progress)
        w.signals.finished.connect(on_finished)
        w.signals.result.connect(on_result)
        w.signals.error.connect(on_error)
        workers.append(w)

    for i in range(5):
        w = GenericWorker(error_fn, i)
        w.signals.started.connect(on_started)
        w.signals.finished.connect(on_finished)
        w.signals.result.connect(on_result)
        w.signals.error.connect(on_error)
        workers.append(w)

    assert len(workers) == 20

    for w in workers:
        pool.start(w)

    start_time = time.time()
    while time.time() - start_time < 5.0:
        QCoreApplication.processEvents()
        with lock:
            if stats["started"] == 20 and (stats["finished"] + stats["error"]) == 20:
                break
        time.sleep(0.01)

    pool.waitForDone(2000)
    for _ in range(20):
        QCoreApplication.processEvents()

    assert stats["started"] == 20, f"Expected 20 started, got {stats['started']}"
    assert stats["finished"] == 15, f"Expected 15 finished, got {stats['finished']}"
    assert stats["result"] == 15, f"Expected 15 result, got {stats['result']}"
    assert stats["error"] == 5, f"Expected 5 error, got {stats['error']}"
    assert stats["progress_count"] == 20, f"Expected 20 progress events, got {stats['progress_count']}"


def test_generic_worker_cancellation_pre_execution(qapp: Any) -> None:
    """Workers cancelled prior to execution must not run or emit lifecycle signals."""
    pool = QThreadPool.globalInstance()
    invoked = False

    def target_fn() -> int:
        nonlocal invoked
        invoked = True
        return 100

    w = GenericWorker(target_fn)
    emissions: list[str] = []
    w.signals.started.connect(lambda: emissions.append("started"))
    w.signals.finished.connect(lambda r: emissions.append("finished"))
    w.signals.error.connect(lambda t, m: emissions.append("error"))

    w.cancel()
    assert w.is_cancelled is True

    pool.start(w)
    pool.waitForDone(1000)
    for _ in range(10):
        QCoreApplication.processEvents()

    assert not invoked, "Worker function executed despite pre-cancellation"
    assert len(emissions) == 0, f"Signals emitted for cancelled worker: {emissions}"


def test_generic_worker_cancellation_mid_execution(qapp: Any) -> None:
    """Workers cancelled mid-execution must suppress finished and result emissions."""
    pool = QThreadPool.globalInstance()
    results: list[Any] = []

    def slow_fn() -> str:
        time.sleep(0.04)
        return "completed"

    w = GenericWorker(slow_fn)
    w.signals.finished.connect(lambda r: results.append(r))
    w.signals.result.connect(lambda r: results.append(r))

    pool.start(w)
    time.sleep(0.01)
    w.cancel()

    pool.waitForDone(1000)
    for _ in range(10):
        QCoreApplication.processEvents()

    assert len(results) == 0, f"Signals were emitted despite cancellation: {results}"


def test_generic_worker_signature_inspection_safety(qapp: Any) -> None:
    """GenericWorker with progress_enabled=True must not crash when fn lacks progress_callback."""
    pool = QThreadPool.globalInstance()
    results: list[int] = []

    def multiply(a: int, b: int) -> int:
        return a * b

    w = GenericWorker(multiply, 7, 8, progress_enabled=True)
    w.signals.result.connect(lambda r: results.append(r))

    pool.start(w)
    pool.waitForDone(1000)
    for _ in range(10):
        QCoreApplication.processEvents()

    assert results == [56]


def test_generic_worker_burst_concurrency(qapp: Any) -> None:
    """Stress test: 40 simultaneous burst workers handle queueing and signal delivery."""
    pool = QThreadPool.globalInstance()
    counts = {"done": 0, "err": 0}
    lock = threading.Lock()

    def on_done(res: Any) -> None:
        with lock:
            counts["done"] += 1

    def on_err(err_t: str, err_m: str) -> None:
        with lock:
            counts["err"] += 1

    def worker_payload(idx: int) -> int:
        if idx % 5 == 0:
            raise RuntimeError(f"error_{idx}")
        return idx * 10

    workers: list[GenericWorker] = []
    for i in range(40):
        w = GenericWorker(worker_payload, i)
        w.signals.finished.connect(on_done)
        w.signals.error.connect(on_err)
        workers.append(w)

    for w in workers:
        pool.start(w)

    pool.waitForDone(3000)
    for _ in range(30):
        QCoreApplication.processEvents()

    assert counts["done"] == 32, f"Expected 32 successful workers, got {counts['done']}"
    assert counts["err"] == 8, f"Expected 8 errored workers, got {counts['err']}"


# ============================================================================
# 2. extract_article Adversarial Tests
# ============================================================================


def test_extract_article_adversarial_schemes_and_inputs() -> None:
    """Adversarial: Non-string inputs, invalid schemes, and malformed URLs return error dicts."""
    bad_inputs = [None, "", "   ", 1234, 3.1415, [], {}, b"http://example.com"]
    for bad in bad_inputs:
        res = extract_article(bad)  # type: ignore[arg-type]
        assert isinstance(res, dict)
        assert res["error"] is not None
        assert res["text"] == ""
        assert res["content"] == ""

    unsupported_schemes = [
        "ftp://example.com/file",
        "file:///etc/shadow",
        "javascript:alert(1)",
        "data:text/html,<h1>XSS</h1>",
        "mailto:test@example.com",
        "gopher://gopher.floodgap.com",
        "ssh://git@github.com",
        "ws://websocket.org",
        "http://",
        "https://",
        "http:///no-host/path",
        "just-a-plain-string",
    ]
    for url in unsupported_schemes:
        res = extract_article(url)
        assert isinstance(res, dict)
        assert res["error"] is not None
        assert res["text"] == ""


def test_extract_article_http_errors_and_empty_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial: 404/500/timeout responses and empty/whitespace bodies are handled cleanly."""
    mock_traf = types.ModuleType("trafilatura")
    mock_traf.extract_metadata = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "trafilatura", mock_traf)

    # 404 / 500: fetch_url returns None
    mock_traf.fetch_url = MagicMock(return_value=None)
    res_404 = extract_article("https://example.com/not-found-404")
    assert res_404["error"] is not None
    assert "HTTP error or empty response" in res_404["error"]
    assert res_404["text"] == ""

    # Network exceptions
    for exc in [ConnectionResetError("Reset"), TimeoutError("Timed out"), OSError("Unreachable")]:
        mock_traf.fetch_url = MagicMock(side_effect=exc)
        res_exc = extract_article("https://example.com/timeout")
        assert res_exc["error"] is not None
        assert "Network error" in res_exc["error"]
        assert res_exc["text"] == ""

    # Empty body
    mock_traf.fetch_url = MagicMock(return_value="<html><body></body></html>")
    mock_traf.extract = MagicMock(return_value=None)
    res_empty = extract_article("https://example.com/empty")
    assert res_empty["error"] is not None
    assert "No extractable text content found" in res_empty["error"]
    assert res_empty["text"] == ""

    mock_traf.extract = MagicMock(return_value="")
    res_empty_str = extract_article("https://example.com/empty-str")
    assert res_empty_str["error"] is not None
    assert "No extractable text content found" in res_empty_str["error"]


def test_extract_article_non_html_and_metadata_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial: JSON, binary null-bytes, and HTML fallback title and language parsing."""
    mock_traf = types.ModuleType("trafilatura")
    monkeypatch.setitem(sys.modules, "trafilatura", mock_traf)

    # JSON response
    mock_traf.fetch_url = MagicMock(return_value='{"name": "test", "items": [1, 2]}')
    mock_traf.extract = MagicMock(return_value='{"name": "test", "items": [1, 2]}')
    mock_traf.extract_metadata = MagicMock(return_value=None)
    res_json = extract_article("https://example.com/data.json")
    assert res_json["error"] is None
    assert "test" in res_json["text"]

    # Raw binary with null bytes
    mock_traf.fetch_url = MagicMock(return_value="binary\x00data\x00stream")
    mock_traf.extract = MagicMock(return_value="binary data stream")
    res_bin = extract_article("https://example.com/bin.dat")
    assert res_bin["error"] is None

    # Title and language extraction from HTML tags
    mock_traf.fetch_url = MagicMock(
        return_value=(
            '<html lang="pt"><head><title>Título da Página</title></head>'
            "<body><h1>Cabeçalho Principal</h1><p>Texto do artigo.</p></body></html>"
        )
    )
    mock_traf.extract = MagicMock(return_value="# Cabeçalho Principal\n\nTexto do artigo.")
    mock_traf.extract_metadata = MagicMock(side_effect=Exception("Metadata failure"))

    res_pt = extract_article("https://example.com/artigo")
    assert res_pt["error"] is None
    assert res_pt["title"] == "Cabeçalho Principal"
    assert res_pt["language"] == "pt"


# ============================================================================
# 3. extract_youtube_transcript Adversarial Tests
# ============================================================================


def test_extract_video_id_adversarial_patterns() -> None:
    """Adversarial: URL patterns, query strings, shorts, embed, live, and invalid inputs."""
    patterns = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?t=10s&v=dQw4w9WgXcQ&feature=shared", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("valid_id_123", "valid_id_123"),
    ]
    for inp, expected in patterns:
        assert extract_video_id(inp) == expected

    for invalid in [None, "", "   ", 1234, [], {}]:
        assert extract_video_id(invalid) == ""  # type: ignore[arg-type]


def test_extract_youtube_transcript_exceptions_and_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial: Transcript API exceptions, missing subtitles, and corrupted timestamps."""
    mock_yt_mod = types.ModuleType("youtube_transcript_api")
    mock_api = MagicMock()
    mock_yt_mod.YouTubeTranscriptApi = mock_api
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mock_yt_mod)

    # Empty inputs
    for bad in [None, "", "   ", 123]:
        assert extract_youtube_transcript(bad) == []  # type: ignore[arg-type]

    # API exceptions
    for exc in [
        RuntimeError("TranscriptsDisabled"),
        KeyError("NoTranscriptFound"),
        ConnectionError("Network down"),
        Exception("429 Too Many Requests"),
    ]:
        mock_api.get_transcript.side_effect = exc
        res = extract_youtube_transcript("dQw4w9WgXcQ")
        assert res == []

    # Non-list payload
    mock_api.get_transcript.side_effect = None
    mock_api.get_transcript.return_value = "not a list"
    assert extract_youtube_transcript("dQw4w9WgXcQ") == []

    # Corrupted fragments, HTML entities, and missing timestamp keys
    mock_api.get_transcript.return_value = [
        {"text": "Hello &amp; welcome &#39;friends&#39;", "start": 1.5, "duration": 2.0},
        {"text": "Missing start and duration"},
        {"text": "String timestamps", "start": "4.5", "duration": "1.5"},
        {"text": "Invalid float timestamp", "start": "corrupt", "duration": None},
        "non-dict item",
    ]
    res_corrupt = extract_youtube_transcript("dQw4w9WgXcQ")
    assert len(res_corrupt) == 4
    assert res_corrupt[0]["text"] == "Hello & welcome 'friends'"
    assert res_corrupt[0]["start"] == 1.5
    assert res_corrupt[1]["start"] == 0.0
    assert res_corrupt[1]["duration"] == 0.0
    assert res_corrupt[2]["start"] == 4.5
    assert res_corrupt[2]["duration"] == 1.5
    assert res_corrupt[3]["start"] == 0.0


def test_transcript_formatting_and_chunking() -> None:
    """Verify transcript to plain text / markdown converters and article chunking."""
    assert transcript_to_text([]) == ""
    assert "*No subtitles available.*" in transcript_to_markdown([])

    fragments = [
        {"text": "First sentence.", "start": 5.0, "duration": 2.0},
        {"text": "Second sentence.", "start": 70.0, "duration": 3.0},
    ]
    assert transcript_to_text(fragments) == "First sentence. Second sentence."
    md = transcript_to_markdown(fragments, title="Test Notes")
    assert "# Test Notes" in md
    assert "- **[00:05]** First sentence." in md
    assert "- **[01:10]** Second sentence." in md

    # Article chunking
    assert chunk_article_text("") == []
    assert chunk_article_text("Short") == ["Short"]
    long_doc = "\n\n".join([f"Para {i}: " + ("word " * 50) for i in range(8)])
    chunks = chunk_article_text(long_doc, max_chunk_size=400)
    assert len(chunks) > 1
    assert "".join(chunks).replace("\n\n", "") == long_doc.replace("\n\n", "")


# ============================================================================
# 4. Concurrent Ingestion in Worker Pool
# ============================================================================


def test_concurrent_ingestion_in_worker_pool(monkeypatch: pytest.MonkeyPatch, qapp: Any) -> None:
    """Stress test: 20 simultaneous ingestion workers running on QThreadPool."""
    mock_traf = types.ModuleType("trafilatura")
    mock_traf.fetch_url = MagicMock(return_value="<html><h1>Title</h1><p>Body</p></html>")
    mock_traf.extract = MagicMock(return_value="# Title\n\nBody")
    mock_traf.extract_metadata = MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "trafilatura", mock_traf)

    mock_yt = types.ModuleType("youtube_transcript_api")

    def mock_get_subtitles(video_id: str, languages: Any = None) -> list[dict[str, Any]]:
        if "bad" in video_id:
            raise RuntimeError(f"VideoUnavailable: {video_id}")
        return [{"text": "Hello world", "start": 0.0, "duration": 1.0}]

    mock_yt_api = MagicMock()
    mock_yt_api.get_transcript = mock_get_subtitles
    mock_yt.YouTubeTranscriptApi = mock_yt_api
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", mock_yt)

    pool = QThreadPool.globalInstance()
    results = {"art_ok": 0, "art_err": 0, "yt_ok": 0, "yt_empty": 0}
    lock = threading.Lock()

    def make_article_worker(url: str, fail: bool) -> GenericWorker:
        def task() -> dict[str, Any]:
            target_url = "ftp://bad-scheme.com" if fail else url
            return extract_article(target_url)

        w = GenericWorker(task)

        def on_res(r: dict[str, Any]) -> None:
            with lock:
                if r.get("error"):
                    results["art_err"] += 1
                else:
                    results["art_ok"] += 1

        w.signals.result.connect(on_res)
        return w

    def make_yt_worker(url: str, fail: bool) -> GenericWorker:
        def task() -> list[dict[str, Any]]:
            target_id = "bad_video_id" if fail else url
            return extract_youtube_transcript(target_id)

        w = GenericWorker(task)

        def on_res(r: list[dict[str, Any]]) -> None:
            with lock:
                if len(r) > 0:
                    results["yt_ok"] += 1
                else:
                    results["yt_empty"] += 1

        w.signals.result.connect(on_res)
        return w

    workers: list[GenericWorker] = []
    for i in range(5):
        workers.append(make_article_worker(f"https://example.com/ok_{i}", fail=False))
    for i in range(5):
        workers.append(make_article_worker(f"https://example.com/err_{i}", fail=True))
    for i in range(5):
        workers.append(make_yt_worker(f"valid_vid_{i}", fail=False))
    for i in range(5):
        workers.append(make_yt_worker(f"bad_vid_{i}", fail=True))

    assert len(workers) == 20

    for w in workers:
        pool.start(w)

    start_time = time.time()
    while time.time() - start_time < 5.0:
        QCoreApplication.processEvents()
        with lock:
            if sum(results.values()) == 20:
                break
        time.sleep(0.01)

    pool.waitForDone(2000)
    for _ in range(20):
        QCoreApplication.processEvents()

    assert results["art_ok"] == 5
    assert results["art_err"] == 5
    assert results["yt_ok"] == 5
    assert results["yt_empty"] == 5

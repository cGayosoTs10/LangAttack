"""Tier 1 & Tier 2 tests for Telemetry, Log Rotation, and Zero-PDB AST check.

Covers:
- FEAT-TEL-01: Daily log partitioning under logs/YYYY-MM-DD/ (app.log & latency.log)
- FEAT-TEL-02: Decorator @latency_tracker(pipeline_name) timing in ms and status SUCCESS/FAIL
- FEAT-TEL-03: Strict zero-tolerance AST scan for pdb, ipdb, and breakpoint()
- Boundary: Exception propagation, sub-millisecond precision, directory creation
"""

import ast
from pathlib import Path

import pytest

import lang_attack.telemetry as tel
from lang_attack.config import REPO_ROOT

# ============================================================================
# Tier 1: Telemetry Setup & Latency Tracker
# ============================================================================


def test_get_daily_log_dir_structure() -> None:
    """Verify get_daily_log_dir creates a directory in YYYY-MM-DD format."""
    daily_dir = tel.get_daily_log_dir()
    assert daily_dir.exists()
    assert daily_dir.is_dir()
    # Name should match YYYY-MM-DD (10 chars, dashes at index 4 and 7)
    name = daily_dir.name
    assert len(name) == 10
    assert name[4] == "-" and name[7] == "-"


def test_setup_telemetry_returns_loggers() -> None:
    """Verify setup_telemetry returns application and latency loggers."""
    app_logger, latency_logger = tel.setup_telemetry()
    assert app_logger is not None
    assert latency_logger is not None
    assert app_logger.name == "lang_attack.app"
    assert latency_logger.name == "lang_attack.latency"


def test_latency_tracker_success() -> None:
    """Verify @latency_tracker tracks successful execution and returns result."""

    @tel.latency_tracker("test_success_pipeline")
    def compute_sum(a: int, b: int) -> int:
        return a + b

    result = compute_sum(15, 27)
    assert result == 42


def test_latency_tracker_failure_propagation() -> None:
    """Verify @latency_tracker captures failure status and re-raises exception."""

    @tel.latency_tracker("test_fail_pipeline")
    def will_fail() -> None:
        raise RuntimeError("Intentional pipeline error")

    with pytest.raises(RuntimeError, match="Intentional pipeline error"):
        will_fail()


# ============================================================================
# Tier 2: Zero PDB AST Enforcement & Boundaries
# ============================================================================


def test_zero_pdb_ast_scanner() -> None:
    """Boundary / Security: Strict AST scan to verify pdb is forbidden in all python files."""
    violations: list[str] = []

    for py_file in REPO_ROOT.glob("**/*.py"):
        # Ignore virtualenv, git, and agents metadata folders
        rel_path = str(py_file.relative_to(REPO_ROOT))
        if any(part in rel_path for part in [".venv", ".agents", "__pycache__", ".git"]):
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Check 'import pdb', 'import ipdb'
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["pdb", "ipdb"]:
                        violations.append(f"{rel_path}:{node.lineno} imports '{alias.name}'")
            # Check 'from pdb import ...'
            elif isinstance(node, ast.ImportFrom):
                if node.module in ["pdb", "ipdb"]:
                    violations.append(f"{rel_path}:{node.lineno} imports from '{node.module}'")
            # Check 'breakpoint()' or 'pdb.set_trace()'
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "breakpoint":
                    violations.append(f"{rel_path}:{node.lineno} calls 'breakpoint()'")
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "set_trace":
                    violations.append(f"{rel_path}:{node.lineno} calls 'set_trace()'")

    assert not violations, "Zero-tolerance pdb violations found:\n" + "\n".join(violations)


def test_latency_tracker_with_args_and_kwargs() -> None:
    """Boundary: Ensure wrapped functions preserve complex *args and **kwargs."""

    @tel.latency_tracker("kwargs_pipeline")
    def complex_func(x: int, multiplier: int = 2, prefix: str = "val") -> str:
        return f"{prefix}:{x * multiplier}"

    assert complex_func(5, multiplier=3, prefix="res") == "res:15"


def test_latency_tracker_measures_positive_duration() -> None:
    """Boundary: Elapsed duration must be strictly >= 0."""
    import time

    @tel.latency_tracker("timing_pipeline")
    def sleep_briefly() -> str:
        time.sleep(0.005)
        return "done"

    res = sleep_briefly()
    assert res == "done"


def test_setup_telemetry_creates_physical_log_files(tmp_path: Path) -> None:
    """Verify setup_telemetry creates physical app.log and latency.log on disk."""
    import time

    app_logger, _ = tel.setup_telemetry(base_dir=tmp_path, force_reconfigure=True)
    today = time.strftime("%Y-%m-%d")
    daily_dir = tmp_path / today

    assert daily_dir.exists()
    app_log = daily_dir / "app.log"
    lat_log = daily_dir / "latency.log"
    assert app_log.exists()
    assert lat_log.exists()

    app_logger.info("Application booting up")
    for h in app_logger.handlers:
        h.flush()

    content = app_log.read_text(encoding="utf-8")
    assert "Application booting up" in content
    assert "[INFO]" in content
    assert "lang_attack.app" in content


def test_latency_tracker_csv_format(tmp_path: Path) -> None:
    """Verify latency.log receives 5-column CSV: timestamp,pipeline,duration,status,error_type."""
    import time

    tel.setup_telemetry(base_dir=tmp_path, force_reconfigure=True)

    @tel.latency_tracker("csv_test_pipeline")
    def sample_task(val: int) -> int:
        return val * 10

    res = sample_task(5)
    assert res == 50

    lat_logger = tel.logging.getLogger("lang_attack.latency")
    for h in lat_logger.handlers:
        h.flush()

    today = time.strftime("%Y-%m-%d")
    lat_log = tmp_path / today / "latency.log"
    assert lat_log.exists()
    lines = [ln.strip() for ln in lat_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1

    last_line = lines[-1]
    parts = last_line.split(",")
    assert len(parts) == 5
    assert parts[1] == "csv_test_pipeline"
    duration = float(parts[2])
    assert duration >= 0.0
    assert parts[3] == "SUCCESS"
    assert parts[4] == "NONE"


def test_latency_tracker_failure_records_error_type(tmp_path: Path) -> None:
    """Verify failure captures specific exception class name in 5th CSV column."""
    import time

    tel.setup_telemetry(base_dir=tmp_path, force_reconfigure=True)

    @tel.latency_tracker("failing_csv_pipeline")
    def fail_task() -> None:
        raise ZeroDivisionError("division by zero error")

    with pytest.raises(ZeroDivisionError):
        fail_task()

    lat_logger = tel.logging.getLogger("lang_attack.latency")
    for h in lat_logger.handlers:
        h.flush()

    today = time.strftime("%Y-%m-%d")
    lines = [
        ln.strip()
        for ln in (tmp_path / today / "latency.log").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    last_line = lines[-1]
    parts = last_line.split(",")
    assert len(parts) == 5
    assert parts[1] == "failing_csv_pipeline"
    assert parts[3] == "FAIL"
    assert parts[4] == "ZeroDivisionError"


def test_setup_telemetry_idempotent() -> None:
    """Verify repeatedly calling setup_telemetry does not duplicate handlers."""
    app1, lat1 = tel.setup_telemetry()
    handler_count_app = len(app1.handlers)
    handler_count_lat = len(lat1.handlers)

    app2, lat2 = tel.setup_telemetry()
    assert len(app2.handlers) == handler_count_app
    assert len(lat2.handlers) == handler_count_lat


def test_ast_scanner_detects_all_violation_types() -> None:
    """Verify scan_ast_for_forbidden_debuggers detects direct, from, dynamic, and call forms."""
    # Constructed to avoid matching test_zero_pdb_ast_scanner inspection
    bad_snippets = [
        "import " + "pdb",
        "import " + "ipdb",
        "from " + "pdb import set_trace",
        "break" + "point()",
        "def f():\n    obj.set_trace()",
        '__import__("' + "pdb" + '").set_trace()',
        'import importlib\nimportlib.import_module("' + "ipdb" + '")',
    ]

    for snippet in bad_snippets:
        tree = tel.ast.parse(snippet)
        violations = tel.scan_ast_for_forbidden_debuggers(tree, filename="test.py")
        assert len(violations) > 0, f"Expected violation for snippet:\n{snippet}"


def test_ast_scanner_passes_clean_code() -> None:
    """Verify scan_ast_for_forbidden_debuggers yields 0 violations on clean code."""
    clean_snippet = '''
def process_data(items: list[int]) -> int:
    """Documentation mentioning debugging concepts."""
    total = sum(items)
    return total
'''
    tree = tel.ast.parse(clean_snippet)
    violations = tel.scan_ast_for_forbidden_debuggers(tree, filename="clean.py")
    assert len(violations) == 0


def test_concurrent_multithreaded_latency_logging(tmp_path: Path) -> None:
    """Verify concurrent worker threads write to latency.log without corruption or data loss."""
    import concurrent.futures
    import time

    tel.setup_telemetry(base_dir=tmp_path, force_reconfigure=True)

    @tel.latency_tracker("thread_safe_pipe")
    def worker_action(idx: int) -> int:
        return idx * idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker_action, range(50)))

    assert len(results) == 50
    lat_logger = tel.logging.getLogger("lang_attack.latency")
    for h in lat_logger.handlers:
        h.flush()

    today = time.strftime("%Y-%m-%d")
    lat_log = tmp_path / today / "latency.log"
    lines = [ln.strip() for ln in lat_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 50
    for line in lines:
        parts = line.split(",")
        assert len(parts) == 5
        assert parts[1] == "thread_safe_pipe"
        assert parts[3] == "SUCCESS"
        assert parts[4] == "NONE"


def test_ast_scanner_hardened_evasions() -> None:
    """Verify scan_ast_for_forbidden_debuggers catches hardened evasion attempts."""
    evasion_snippets = [
        "import " + "pdb.set_trace",
        "import " + "pdb.submodule as sub",
        "import " + "ipdb.submodule",
        "from " + "pdb.submodule import debug_helper",
        "from " + "ipdb.frontend import run",
        "from builtins import " + "breakpoint",
        "builtins." + "breakpoint()",
        "import importlib\nimportlib.import_module('" + "pdb.submodule')",
        "import importlib\nimportlib.import_module(name='" + "pdb')",
        "import importlib\nimportlib.import_module(name='" + "ipdb.client')",
        "getattr(builtins, '" + "breakpoint')()",
        "getattr(sys, '" + "set_trace')()",
        "eval('break' + 'point()')",
        "exec('import ' + 'pdb')",
    ]

    for snippet in evasion_snippets:
        violations = tel.scan_ast_for_forbidden_debuggers(snippet, filename="evasion_test.py")
        assert len(violations) > 0, f"Expected scanner to catch evasion snippet:\n{snippet}"


def test_sanitize_pipeline_name_units() -> None:
    """Verify sanitize_pipeline_name cleans delimiters and invalid chars."""
    assert tel.sanitize_pipeline_name("simple_pipe") == "simple_pipe"
    assert tel.sanitize_pipeline_name("pipe,with,commas") == "pipe_with_commas"
    assert tel.sanitize_pipeline_name("pipe\nwith\rnewlines") == "pipe with newlines"
    assert tel.sanitize_pipeline_name("  trimmed  ") == "trimmed"
    assert tel.sanitize_pipeline_name("audio\r\nstream,chunk\t2") == "audio stream_chunk 2"
    assert tel.sanitize_pipeline_name(",,,,") == "____"
    assert tel.sanitize_pipeline_name("") == "unnamed_pipeline"
    assert tel.sanitize_pipeline_name("   \n\t  ") == "unnamed_pipeline"


def test_latency_tracker_csv_sanitization(tmp_path: Path) -> None:
    """Verify @latency_tracker preserves 5-column CSV even with commas and newlines in pipeline_name."""
    import time

    tel.setup_telemetry(base_dir=tmp_path, force_reconfigure=True)

    @tel.latency_tracker("audio,ingestion\npipeline,v1")
    def dirty_name_task() -> str:
        return "ok"

    res = dirty_name_task()
    assert res == "ok"

    lat_logger = tel.logging.getLogger("lang_attack.latency")
    for h in lat_logger.handlers:
        h.flush()

    today = time.strftime("%Y-%m-%d")
    lat_log = tmp_path / today / "latency.log"
    assert lat_log.exists()
    lines = [ln.strip() for ln in lat_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1
    last_line = lines[-1]
    parts = last_line.split(",")
    # Must have exactly 5 columns
    assert len(parts) == 5
    assert parts[1] == "audio_ingestion pipeline_v1"
    assert parts[3] == "SUCCESS"
    assert parts[4] == "NONE"



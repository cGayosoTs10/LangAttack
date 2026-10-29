"""Basic sanity test for repository foundation."""

import importlib

import pytest

import lang_attack
import lang_attack.app
import lang_attack.telemetry
from lang_attack.config import DATABASE_DIR, DEFAULT_DB_PATH, LOGS_DIR, NOTES_DIR, REPO_ROOT


def test_package_metadata() -> None:
    """Verify package initializes with correct version."""
    assert lang_attack.__version__ == "0.1.0"


def test_config_paths() -> None:
    """Verify core configuration paths are resolved properly."""
    assert REPO_ROOT.exists()
    assert DATABASE_DIR.name == "database"
    assert LOGS_DIR.name == "logs"
    assert NOTES_DIR.name == "notes"
    assert DEFAULT_DB_PATH.name == "lang_attack.db"


def test_all_subpackages_importable() -> None:
    """Verify all skeleton modular packages can be imported successfully."""
    packages = [
        "lang_attack.db",
        "lang_attack.schemas",
        "lang_attack.core",
        "lang_attack.workers",
        "lang_attack.ui",
    ]
    for pkg in packages:
        module = importlib.import_module(pkg)
        assert module is not None


def test_app_main() -> None:
    """Verify main entry point function returns zero exit code."""
    assert lang_attack.app.main() == 0


def test_telemetry_stubs() -> None:
    """Verify telemetry setup and latency tracker execution."""
    app_logger, latency_logger = lang_attack.telemetry.setup_telemetry()
    assert app_logger is not None
    assert latency_logger is not None

    daily_dir = lang_attack.telemetry.get_daily_log_dir()
    assert daily_dir.exists()

    @lang_attack.telemetry.latency_tracker("test_pipeline")
    def sample_func(x: int) -> int:
        return x * 2

    assert sample_func(21) == 42


def test_telemetry_tracker_propagates_exception() -> None:
    """Verify latency tracker captures errors and re-raises exceptions."""

    @lang_attack.telemetry.latency_tracker("failing_pipeline")
    def failing_func() -> None:
        raise ValueError("simulated pipeline error")

    with pytest.raises(ValueError, match="simulated pipeline error"):
        failing_func()


def test_no_pdb_in_codebase() -> None:
    """Verify zero tolerance for pdb in all python source files."""
    for py_file in REPO_ROOT.glob("lang_attack/**/*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "import pdb" not in content, f"Found pdb import in {py_file}"
        assert "pdb.set_trace" not in content, f"Found pdb.set_trace in {py_file}"
        assert "breakpoint()" not in content, f"Found breakpoint() in {py_file}"

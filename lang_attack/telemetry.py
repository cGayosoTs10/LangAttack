"""Telemetry, log partitioning, and latency tracking subsystem for LangAttack.

Implements:
- FEAT-TEL-01: Daily log partitioning under logs/YYYY-MM-DD/ (app.log and latency.log)
- FEAT-TEL-02: Decorator @latency_tracker(pipeline_name) recording duration and status
- FEAT-TEL-03: Reusable AST scanner enforcing zero-tolerance prohibition of debuggers
"""

import ast
import logging
import threading
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from lang_attack.config import LOGS_DIR, REPO_ROOT

T = TypeVar("T")

# Constants for AST scanner verification
FORBIDDEN_MODULES: tuple[str, ...] = ("pdb", "ipdb", "pudb", "rpdb")
FORBIDDEN_CALLS: tuple[str, ...] = ("breakpoint",)
FORBIDDEN_ATTRS: tuple[str, ...] = ("set_trace", "post_mortem", "pm")


def _is_forbidden_module(name: str | None) -> bool:
    """Check if a module name is a forbidden debugger or a submodule of one."""
    if not name or not isinstance(name, str):
        return False
    return any(
        name == mod or name.startswith(f"{mod}.")
        for mod in FORBIDDEN_MODULES
    )


def _get_constant_str(node: ast.AST | None) -> str | None:
    """Extract string value from Constant or folded BinOp string concatenations."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _get_constant_str(node.left)
        right = _get_constant_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _extract_module_arg(node: ast.Call) -> str | None:
    """Extract constant module name from positional or keyword arguments."""
    if node.args:
        val = _get_constant_str(node.args[0])
        if val is not None:
            return val
    for kw in node.keywords:
        if kw.arg == "name":
            val = _get_constant_str(kw.value)
            if val is not None:
                return val
    return None


def sanitize_pipeline_name(name: str) -> str:
    """Sanitize pipeline name to ensure 5-column CSV format integrity.

    Replaces commas with underscores, replaces carriage returns and newlines with spaces,
    collapses multiple whitespace characters, and strips leading/trailing whitespace.
    """
    if not isinstance(name, str):
        name = str(name)
    clean = name.replace(",", "_").replace("\r", " ").replace("\n", " ").strip()
    clean = " ".join(clean.split())
    return clean or "unnamed_pipeline"



def get_daily_log_dir(base_dir: Path | None = None) -> Path:
    """Return the current daily log directory under logs/YYYY-MM-DD/.

    Creates the directory if it does not already exist.
    """
    root = base_dir if base_dir is not None else LOGS_DIR
    daily_dir = root / time.strftime("%Y-%m-%d")
    daily_dir.mkdir(parents=True, exist_ok=True)
    return daily_dir


class DailyPartitionFileHandler(logging.Handler):
    """Logging handler that partitions logs into date-based directories (logs/YYYY-MM-DD/).

    Automatically rolls over to a new folder when the date changes across midnight,
    with full thread-safety for multi-worker concurrency.
    """

    def __init__(
        self,
        filename: str,
        base_dir: Path | None = None,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self.filename = filename
        self.base_dir = base_dir
        self.encoding = encoding
        self._current_date: str | None = None
        self._inner_handler: logging.FileHandler | None = None
        self._lock = threading.RLock()

    def _get_inner_handler(self) -> logging.FileHandler:
        today = time.strftime("%Y-%m-%d")
        if self._inner_handler is None or self._current_date != today:
            with self._lock:
                if self._inner_handler is None or self._current_date != today:
                    if self._inner_handler is not None:
                        self._inner_handler.flush()
                        self._inner_handler.close()
                    daily_dir = get_daily_log_dir(self.base_dir)
                    target_file = daily_dir / self.filename
                    self._inner_handler = logging.FileHandler(target_file, encoding=self.encoding)
                    if self.formatter:
                        self._inner_handler.setFormatter(self.formatter)
                    self._current_date = today
        return self._inner_handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            handler = self._get_inner_handler()
            handler.emit(record)
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        with self._lock:
            if self._inner_handler is not None:
                self._inner_handler.flush()

    def close(self) -> None:
        with self._lock:
            if self._inner_handler is not None:
                self._inner_handler.flush()
                self._inner_handler.close()
                self._inner_handler = None
        super().close()


def setup_telemetry(
    base_dir: Path | None = None,
    force_reconfigure: bool = False,
) -> tuple[logging.Logger, logging.Logger]:
    """Initialize and return application and latency loggers with daily partitioned file handlers.

    Idempotent by default: does not attach duplicate handlers if already configured.
    """
    app_logger = logging.getLogger("lang_attack.app")
    latency_logger = logging.getLogger("lang_attack.latency")

    app_logger.setLevel(logging.INFO)
    latency_logger.setLevel(logging.INFO)

    app_logger.propagate = False
    latency_logger.propagate = False

    # Check or reconfigure app_logger
    has_app_handler = any(
        isinstance(h, DailyPartitionFileHandler) and h.filename == "app.log"
        for h in app_logger.handlers
    )
    if force_reconfigure or not has_app_handler:
        if force_reconfigure:
            for h in list(app_logger.handlers):
                app_logger.removeHandler(h)
                h.close()
        app_handler = DailyPartitionFileHandler("app.log", base_dir=base_dir)
        app_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        app_logger.addHandler(app_handler)
        # Ensure physical file is initialized immediately
        app_handler._get_inner_handler()

    # Check or reconfigure latency_logger
    has_latency_handler = any(
        isinstance(h, DailyPartitionFileHandler) and h.filename == "latency.log"
        for h in latency_logger.handlers
    )
    if force_reconfigure or not has_latency_handler:
        if force_reconfigure:
            for h in list(latency_logger.handlers):
                latency_logger.removeHandler(h)
                h.close()
        latency_handler = DailyPartitionFileHandler("latency.log", base_dir=base_dir)
        latency_handler.setFormatter(logging.Formatter("%(message)s"))
        latency_logger.addHandler(latency_handler)
        # Ensure physical file is initialized immediately
        latency_handler._get_inner_handler()

    return app_logger, latency_logger


def setup_daily_logging(
    base_dir: Path | None = None,
) -> tuple[logging.Logger, logging.Logger]:
    """Alias for setup_telemetry to match Section 3 specification."""
    return setup_telemetry(base_dir=base_dir)


def latency_tracker(pipeline_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to measure and log execution latency of critical pipelines.

    Logs CSV format: timestamp,pipeline_name,duration_ms,status,error_type
    Transparently propagates all exceptions without swallow.
    Sanitizes pipeline_name to preserve 5-column CSV format integrity.
    """
    clean_pipeline_name = sanitize_pipeline_name(pipeline_name)

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            status = "SUCCESS"
            error_type = "NONE"
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                status = "FAIL"
                error_type = sanitize_pipeline_name(type(exc).__name__)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                now_str = time.strftime("%Y-%m-%d %H:%M:%S")
                logger = logging.getLogger("lang_attack.latency")
                logger.info(
                    "%s,%s,%.2f,%s,%s",
                    now_str,
                    clean_pipeline_name,
                    elapsed_ms,
                    status,
                    error_type,
                )

        return wrapper

    return decorator


def scan_ast_for_forbidden_debuggers(
    source_or_tree: str | ast.AST,
    filename: str = "<unknown>",
) -> list[str]:
    """Scan source code or AST tree for prohibited debuggers.

    Returns a list of violation descriptions.
    """
    if isinstance(source_or_tree, str):
        try:
            tree = ast.parse(source_or_tree, filename=filename)
        except SyntaxError:
            return [f"{filename}: SyntaxError parsing source code"]
    else:
        tree = source_or_tree

    violations: list[str] = []
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    violations.append(
                        f"{filename}:{lineno}: Prohibited import of '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            if _is_forbidden_module(node.module):
                violations.append(
                    f"{filename}:{lineno}: Prohibited import from '{node.module}'"
                )
            for alias in node.names:
                if alias.name in FORBIDDEN_CALLS or alias.name in FORBIDDEN_ATTRS:
                    violations.append(
                        f"{filename}:{lineno}: Prohibited import of '{alias.name}' from '{node.module}'"
                    )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_id = node.func.id
                if func_id in FORBIDDEN_CALLS or func_id in FORBIDDEN_ATTRS:
                    violations.append(
                        f"{filename}:{lineno}: Prohibited call '{func_id}()'"
                    )
                elif func_id in ("__import__", "import_module"):
                    mod_arg = _extract_module_arg(node)
                    if _is_forbidden_module(mod_arg):
                        violations.append(
                            f"{filename}:{lineno}: Prohibited dynamic import '{func_id}(\"{mod_arg}\")'"
                        )
                elif func_id in ("eval", "exec") and node.args:
                    code_str = _get_constant_str(node.args[0])
                    if code_str:
                        inner = scan_ast_for_forbidden_debuggers(code_str, filename=filename)
                        if inner:
                            violations.append(
                                f"{filename}:{lineno}: Prohibited debugger in dynamic '{func_id}()'"
                            )
                elif func_id == "getattr" and len(node.args) >= 2:
                    second_arg_val = _get_constant_str(node.args[1])
                    if second_arg_val and (second_arg_val in FORBIDDEN_ATTRS or second_arg_val in FORBIDDEN_CALLS):
                        violations.append(
                            f"{filename}:{lineno}: Prohibited dynamic attribute access 'getattr(..., \"{second_arg_val}\")'"
                        )
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in FORBIDDEN_ATTRS or attr_name in FORBIDDEN_CALLS:
                    violations.append(
                        f"{filename}:{lineno}: Prohibited call '{attr_name}()'"
                    )
                elif isinstance(node.func.value, ast.Name) and _is_forbidden_module(node.func.value.id):
                    violations.append(
                        f"{filename}:{lineno}: Prohibited call on forbidden module '{node.func.value.id}.{attr_name}()'"
                    )
                elif attr_name in ("import_module", "__import__"):
                    mod_arg = _extract_module_arg(node)
                    if _is_forbidden_module(mod_arg):
                        violations.append(
                            f"{filename}:{lineno}: Prohibited dynamic import '{attr_name}(\"{mod_arg}\")'"
                        )
    return violations


def scan_codebase_for_debuggers(
    root_dir: Path | None = None,
    exclude_dirs: tuple[str, ...] = (".venv", ".agents", "__pycache__", ".git"),
) -> dict[str, list[str]]:
    """Scan all Python files under root_dir for prohibited debuggers.

    Returns a dictionary mapping relative file paths to lists of violations.
    """
    target_root = root_dir if root_dir is not None else REPO_ROOT
    violations: dict[str, list[str]] = {}

    for py_file in target_root.glob("**/*.py"):
        rel_path = str(py_file.relative_to(target_root))
        if any(part in rel_path for part in exclude_dirs):
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue

        file_violations = scan_ast_for_forbidden_debuggers(tree, filename=rel_path)
        if file_violations:
            violations[rel_path] = file_violations

    return violations


__all__ = [
    "DailyPartitionFileHandler",
    "get_daily_log_dir",
    "latency_tracker",
    "sanitize_pipeline_name",
    "scan_ast_for_forbidden_debuggers",
    "scan_codebase_for_debuggers",
    "setup_daily_logging",
    "setup_telemetry",
]

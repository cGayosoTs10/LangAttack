"""Generic background worker and Qt signals for decoupled GUI concurrency.

Follows the QRunnable + QThreadPool pattern to ensure zero UI thread blocking
during audio synthesis, AI inference, web scraping, and database persistence.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure system PySide6 location is discoverable if not in active venv
try:
    from PySide6.QtCore import QObject, QRunnable, Signal
except ImportError:
    for candidate in [
        f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
        str(
            Path.home()
            / f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        ),
        "/usr/lib/python3.14/site-packages",
    ]:
        if os.path.exists(candidate) and candidate not in sys.path:
            sys.path.append(candidate)
    from PySide6.QtCore import QObject, QRunnable, Signal

logger = logging.getLogger("lang_attack.workers")

__all__ = ["GenericWorker", "WorkerSignals"]


class WorkerSignals(QObject):
    """Qt signals for background worker lifecycle and event communication.

    Signals:
        started: Emitted immediately before task execution begins in worker thread.
        progress: Emitted with (percentage: int, message: str) during task execution.
        finished: Emitted on successful completion with the return value (object).
        result: Dual compatibility signal emitted on successful completion.
        error: Emitted when an exception occurs with (error_type: str, error_message: str).
    """

    started = Signal()
    progress = Signal(int, str)
    finished = Signal(object)
    result = Signal(object)
    error = Signal(str, str)


class GenericWorker(QRunnable):
    """Reusable QRunnable worker executing arbitrary callables on QThreadPool.

    Decouples heavy background computations (AI inference, audio recording,
    web scraping, database transactions) from the PySide6 GUI event loop.
    All status, progress, and results are communicated back across threads
    via WorkerSignals using Qt queued connections.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize worker with callable and invocation arguments.

        Args:
            fn: Target callable to execute in background thread.
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn. Special optional
                keyword arguments:
                - progress_enabled (bool): When True, injects a progress_callback
                  callable into fn (if accepted by fn). Defaults to False.
                - auto_delete (bool): Sets QRunnable autoDelete flag.
                  Defaults to False to prevent Shiboken premature C++ deletion
                  when worker references are retained in Python.
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.progress_enabled: bool = bool(kwargs.pop("progress_enabled", False))
        auto_delete: bool = bool(kwargs.pop("auto_delete", False))
        self.setAutoDelete(auto_delete)
        self.kwargs: dict[str, Any] = kwargs

        self.signals = WorkerSignals()
        self._is_cancelled: bool = False

    def cancel(self) -> None:
        """Request cooperative cancellation of the worker task."""
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check whether cancellation was requested."""
        return self._is_cancelled

    def _emit_progress(self, value: int, message: str = "") -> None:
        """Progress callback helper invoked from payload to emit Qt signal."""
        if not self._is_cancelled:
            self.signals.progress.emit(int(value), str(message))

    def run(self) -> None:
        """Execute the wrapped callable in background thread from thread pool."""
        if self._is_cancelled:
            logger.info("GenericWorker cancelled before execution: %s", self.fn)
            return

        call_kwargs = dict(self.kwargs)
        if self.progress_enabled:
            try:
                sig = inspect.signature(self.fn)
                params = sig.parameters
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                if "progress_callback" in params or has_var_keyword:
                    call_kwargs["progress_callback"] = self._emit_progress
            except (ValueError, TypeError):
                call_kwargs["progress_callback"] = self._emit_progress

        self.signals.started.emit()
        try:
            res = self.fn(*self.args, **call_kwargs)
        except Exception as exc:
            logger.error(
                "GenericWorker execution failed in %s: %s: %s",
                getattr(self.fn, "__name__", str(self.fn)),
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self.signals.error.emit(type(exc).__name__, str(exc))
        else:
            if not self._is_cancelled:
                self.signals.result.emit(res)
                self.signals.finished.emit(res)

"""Telemetry and logging subsystem for LangAttack."""

import logging
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from lang_attack.config import LOGS_DIR

T = TypeVar("T")


def get_daily_log_dir() -> Path:
    """Return the current daily log directory."""
    daily_dir = LOGS_DIR / time.strftime("%Y-%m-%d")
    daily_dir.mkdir(parents=True, exist_ok=True)
    return daily_dir


def setup_telemetry() -> tuple[logging.Logger, logging.Logger]:
    """Initialize and return application and latency loggers."""
    app_logger = logging.getLogger("lang_attack.app")
    latency_logger = logging.getLogger("lang_attack.latency")
    return app_logger, latency_logger


def latency_tracker(pipeline_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to measure and log execution latency of critical pipelines."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            status = "SUCCESS"
            try:
                return fn(*args, **kwargs)
            except Exception:
                status = "FAIL"
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logging.getLogger("lang_attack.latency").info(
                    "%s,%s,%.2f,%s",
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    pipeline_name,
                    elapsed_ms,
                    status,
                )

        return wrapper

    return decorator

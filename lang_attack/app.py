"""Application entry point for LangAttack PySide6 desktop suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from lang_attack.config import REPO_ROOT
from lang_attack.db.connection import init_db
from lang_attack.telemetry import setup_daily_logging
from lang_attack.ui.main_window import MainWindow


def main() -> int:
    """Initialize application subsystems and start Qt event loop."""
    # 1. Initialize daily log partitioning
    setup_daily_logging(REPO_ROOT)

    # 2. Initialize SQLite WAL database & DDL tables
    init_db()

    # 3. Create or get Qt Application
    app = QApplication.instance()
    is_test_run = app is not None or "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("LangAttack")
    app.setOrganizationName("LangAttack")

    # 4. Load dark theme stylesheet
    qss_path = Path(__file__).parent / "ui" / "styles.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    # 5. Create and display main workstation window
    window = MainWindow()
    window.show()

    if is_test_run:
        return 0

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

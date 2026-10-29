"""MainWindow: Primary desktop window with 3-pane horizontal QSplitter layout."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from lang_attack.ui.panels.center_deck import CenterDeck
from lang_attack.ui.panels.left_workspace import LeftWorkspace
from lang_attack.ui.panels.right_lab import RightLab

logger = logging.getLogger("lang_attack.ui.main")


class MainWindow(QMainWindow):
    """Main desktop application window hosting the 3-panel learning workstation."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("LangAttack — Estúdio Inteligente de Aprendizado de Idiomas")
        self.resize(1280, 800)

        self._init_ui()
        self._wire_signals()

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # 3-Pane Horizontal Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Panel 1: Left Workspace (File tree, context selector, URL import)
        self.left_workspace = LeftWorkspace(parent=self.splitter)
        self.splitter.addWidget(self.left_workspace)

        # Panel 2: Center Deck (Markdown reader/editor, voice chat, exercises)
        self.center_deck = CenterDeck(parent=self.splitter)
        self.splitter.addWidget(self.center_deck)

        # Panel 3: Right Lab (Speech recording, playback, GOP phonetic visualizer)
        self.right_lab = RightLab(parent=self.splitter)
        self.splitter.addWidget(self.right_lab)

        # Set balanced initial proportions: Left ~240px, Center ~640px, Right ~400px
        self.splitter.setSizes([240, 640, 400])

        main_layout.addWidget(self.splitter)

    def _wire_signals(self) -> None:
        """Wire inter-panel reactive events."""
        # When a note is opened in left workspace, load into editor
        self.left_workspace.note_selected.connect(self.center_deck.load_note)

        # When a sentence is chosen for speech training in center deck, send to right lab
        self.center_deck.sentence_to_practice.connect(self.right_lab.set_target_text)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Gracefully handle window close: cleanup panels and wait for background threads."""
        logger.info("MainWindow closing, initiating graceful teardown...")
        if hasattr(self, "right_lab") and self.right_lab is not None:
            try:
                self.right_lab.cleanup()
            except Exception as e:
                logger.warning("Error during right_lab cleanup: %s", e)

        try:
            pool = QThreadPool.globalInstance()
            pool.clear()
            pool.waitForDone(1000)
        except Exception as e:
            logger.warning("Error waiting for background threads: %s", e)

        event.accept()
        super().closeEvent(event)


__all__ = ["MainWindow"]


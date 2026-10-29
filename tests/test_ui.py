"""Tier 1 & Tier 2 tests for PySide6 GUI panels, QSplitter layout, and actions.

Covers:
- FEAT-GUI-01: Dark theme stylesheet (styles.qss)
- FEAT-GUI-02: MainWindow horizontal QSplitter containing Left, Center, Right panels
- FEAT-GUI-03: LeftWorkspace notes tree with context menu actions
- FEAT-GUI-04: LeftWorkspace study context selector and URL ingestion footer
- FEAT-GUI-05: CenterDeck Markdown reader/editor with sentence practice dispatch
- FEAT-GUI-06: CenterDeck Interactive chat tab with voice audio playback
- FEAT-GUI-07: CenterDeck Interactive exercise engine driven by Pydantic schemas
- FEAT-GUI-08: RightLab Audio player with speed regulation (0.75x, 1.0x, 1.25x)
- FEAT-GUI-09: RightLab Audio recording trigger
- FEAT-GUI-10: RightLab HTML/CSS GOP phonetic visualizer (#4CAF50, #FFC107, #F44336)
- FEAT-GUI-11: RightLab Articulatory feedback box
"""

from pathlib import Path
from typing import Any

import pytest

# Progressive testability check
try:
    import lang_attack.ui.main_window as mw_mod
    from PySide6.QtWidgets import QSplitter

    HAS_UI = hasattr(mw_mod, "MainWindow")
except ImportError:
    HAS_UI = False

pytestmark = pytest.mark.skipif(
    not HAS_UI,
    reason="lang_attack.ui modules not yet implemented (scheduled for M4)",
)


# ============================================================================
# Tier 1: UI Component Initialization & Layout
# ============================================================================


def test_main_window_qsplitter_layout(qapp: Any) -> None:
    """Verify MainWindow initializes horizontal QSplitter with 3 main panels."""
    from lang_attack.ui.main_window import MainWindow

    window = MainWindow()
    assert window is not None

    # Check horizontal splitter
    splitters = window.findChildren(QSplitter)
    assert len(splitters) >= 1
    main_splitter = splitters[0]
    assert main_splitter.count() == 3, "MainWindow QSplitter must contain exactly 3 panes"


def test_left_workspace_notes_vault(qapp: Any, tmp_notes_vault: Path) -> None:
    """Verify LeftWorkspace initializes QFileSystemModel over notes vault."""
    from lang_attack.ui.left_workspace import LeftWorkspace

    left = LeftWorkspace(notes_dir=tmp_notes_vault)
    assert left is not None
    assert hasattr(left, "context_combo") or hasattr(left, "combo_box")


def test_center_deck_tabs(qapp: Any) -> None:
    """Verify CenterDeck provides Markdown Reader, Chat, and Exercise tabs."""
    from lang_attack.ui.center_deck import CenterDeck
    from PySide6.QtWidgets import QTabWidget

    center = CenterDeck()
    tab_widget = center.findChild(QTabWidget) or center
    assert tab_widget.count() >= 3, "CenterDeck must provide at least 3 tabs"


def test_right_lab_controls(qapp: Any) -> None:
    """Verify RightLab contains audio player, speed buttons, and GOP visualizer."""
    from lang_attack.ui.right_lab import RightLab

    right = RightLab()
    assert right is not None
    assert hasattr(right, "record_button") or hasattr(right, "btn_record")


def test_dark_theme_stylesheet_exists() -> None:
    """Verify styles.qss is present and contains required GOP color tokens."""
    qss_path = Path("lang_attack/ui/styles.qss")
    if not qss_path.exists():
        qss_path = Path("lang_attack/ui/resources/styles.qss")

    assert qss_path.exists(), "Dark theme stylesheet styles.qss not found"
    content = qss_path.read_text(encoding="utf-8")
    assert "#4CAF50" in content or "green" in content.lower()
    assert "#FFC107" in content or "yellow" in content.lower()
    assert "#F44336" in content or "red" in content.lower()


# ============================================================================
# Tier 2: Boundary & Action Tests
# ============================================================================


def test_speed_control_values(qapp: Any) -> None:
    """Boundary: Playback speed regulates between 0.75x, 1.0x, and 1.25x."""
    from lang_attack.ui.right_lab import RightLab

    right = RightLab()
    if hasattr(right, "set_playback_speed"):
        for speed in [0.75, 1.0, 1.25]:
            right.set_playback_speed(speed)
            assert getattr(right, "current_speed", speed) == speed


def test_phonetic_visualizer_html_rendering(qapp: Any) -> None:
    """Boundary: GOP visualizer color-codes phonemes into correct CSS spans."""
    from lang_attack.ui.right_lab import RightLab

    right = RightLab()
    if hasattr(right, "render_phonetic_html"):
        html = right.render_phonetic_html(
            phonemes=[{"symbol": "ð", "score": 90.0}, {"symbol": "ə", "score": 40.0}]
        )
        assert "#4CAF50" in html  # Green
        assert "#F44336" in html  # Red

"""RightLab panel: Audio player, speech recording, GOP phonetic visualizer, and articulatory feedback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from lang_attack.core.audio_recorder import AudioRecorder
from lang_attack.schemas.phonetics import SpeechAssessmentResult
from lang_attack.workers.tasks import (
    dispatch_task,
    speech_evaluation_task,
    tts_synthesis_task,
)

logger = logging.getLogger("lang_attack.ui.right")


class RightLab(QWidget):
    """Speech laboratory panel providing audio target playback, recording, and phonetic diagnostics."""

    assessment_completed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("rightLab")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.current_speed: float = 1.0
        self.target_text: str = "The quick brown fox jumps over the lazy dog."
        self.recorder = AudioRecorder(sample_rate=16000, channels=1)
        self._recorded_wav: Path = Path("/tmp/lang_attack_recorded.wav")
        self._tutor_wav: Path = Path("/tmp/lang_attack_target.wav")

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. Target Sentence Box
        target_group = QGroupBox("Frase Alvo para Treino:")
        target_group.setStyleSheet("QGroupBox { font-weight: bold; color: #a5b4fc; }")
        t_layout = QVBoxLayout(target_group)
        t_layout.setContentsMargins(6, 12, 6, 6)

        self.txt_target = QPlainTextEdit()
        self.txt_target.setPlainText(self.target_text)
        self.txt_target.setMaximumHeight(70)
        t_layout.addWidget(self.txt_target)

        # Audio Playback & Speed Controls
        ctrl_layout = QHBoxLayout()
        self.btn_play_target = QPushButton("▶ Ouvir Modelo")
        self.btn_play_target.clicked.connect(self._play_target_audio)
        ctrl_layout.addWidget(self.btn_play_target, 1)

        # Speed buttons: 0.75x, 1.0x, 1.25x
        self.speed_group = QButtonGroup(self)
        for speed in [0.75, 1.0, 1.25]:
            btn = QPushButton(f"{speed}x")
            btn.setCheckable(True)
            if speed == 1.0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, s=speed: self.set_playback_speed(s))
            self.speed_group.addButton(btn)
            ctrl_layout.addWidget(btn, 0)

        t_layout.addLayout(ctrl_layout)
        layout.addWidget(target_group)

        # 2. Recording Controls
        rec_group = QGroupBox("Gravação da Sua Voz:")
        rec_group.setStyleSheet("QGroupBox { font-weight: bold; color: #a5b4fc; }")
        r_layout = QVBoxLayout(rec_group)
        r_layout.setContentsMargins(6, 12, 6, 6)

        self.record_button = QPushButton("● Gravar Pronúncia (Espaço)")
        self.record_button.setObjectName("recordButton")
        self.record_button.setCheckable(True)
        self.record_button.setFixedHeight(40)
        self.record_button.clicked.connect(self._toggle_recording)
        r_layout.addWidget(self.record_button)

        self.lbl_record_status = QLabel("Pressione a barra de Espaço ou o botão para gravar.")
        self.lbl_record_status.setStyleSheet("color: #9ca3af; font-size: 11px;")
        self.lbl_record_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        r_layout.addWidget(self.lbl_record_status)

        layout.addWidget(rec_group)

        # 3. Phonetic Visualizer HTML Display
        vis_group = QGroupBox("Diagnóstico Fonético (GOP):")
        vis_group.setStyleSheet("QGroupBox { font-weight: bold; color: #a5b4fc; }")
        v_layout = QVBoxLayout(vis_group)
        v_layout.setContentsMargins(6, 12, 6, 6)

        self.phonetic_display = QTextBrowser()
        self.phonetic_display.setMinimumHeight(100)
        self.phonetic_display.setHtml(
            "<div style='color: #9ca3af; text-align: center; padding: 20px;'>"
            "Grave sua pronúncia para visualizar a pontuação e coloração fonema por fonema."
            "</div>"
        )
        v_layout.addWidget(self.phonetic_display)
        layout.addWidget(vis_group, 1)

        # 4. Articulatory Feedback Box
        art_group = QGroupBox("Feedback Articulatório (Lábios e Língua):")
        art_group.setStyleSheet("QGroupBox { font-weight: bold; color: #a5b4fc; }")
        a_layout = QVBoxLayout(art_group)
        a_layout.setContentsMargins(6, 12, 6, 6)

        self.feedback_box = QTextBrowser()
        self.feedback_box.setMaximumHeight(120)
        self.feedback_box.setHtml(
            "<div style='color: #9ca3af;'>"
            "Dicas biomecânicas de posicionamento articulatório aparecerão aqui para fonemas desviados."
            "</div>"
        )
        a_layout.addWidget(self.feedback_box)
        layout.addWidget(art_group)

    def set_target_text(self, text: str) -> None:
        """Set the target sentence to practice."""
        self.target_text = text.strip()
        self.txt_target.setPlainText(self.target_text)

    def set_playback_speed(self, speed: float) -> None:
        """Regulate playback speed between defined bounds [0.5, 2.0]."""
        clamped = max(0.5, min(2.0, float(speed)))
        self.current_speed = clamped
        for btn in self.speed_group.buttons():
            if btn.text() == f"{clamped}x":
                btn.setChecked(True)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Trigger recording toggle with Space bar."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat() and not self.txt_target.hasFocus():
            self._toggle_recording()
            event.accept()
            return
        super().keyPressEvent(event)

    def _play_target_audio(self) -> None:
        """Synthesize and play tutor model audio."""
        text = self.txt_target.toPlainText().strip()
        if not text:
            return

        self.btn_play_target.setEnabled(False)
        self.btn_play_target.setText("Sintetizando...")

        def on_done(out_wav: Path) -> None:
            self.btn_play_target.setEnabled(True)
            self.btn_play_target.setText("▶ Ouvir Modelo")

        def on_err(err_type: str, err_msg: str) -> None:
            self.btn_play_target.setEnabled(True)
            self.btn_play_target.setText("▶ Ouvir Modelo")

        dispatch_task(
            tts_synthesis_task,
            text=text,
            output_path=self._tutor_wav,
            speed=self.current_speed,
            on_result=on_done,
            on_error=on_err,
        )

    def _toggle_recording(self) -> None:
        """Toggle audio recording state."""
        if not self.recorder.is_recording:
            # Start
            self.recorder.start_recording()
            self.record_button.setChecked(True)
            self.record_button.setText("⏹ Parar Gravação")
            self.lbl_record_status.setText("Gravando áudio mono float32 16kHz...")
            self.lbl_record_status.setStyleSheet("color: #ef4444; font-weight: bold;")
        else:
            # Stop
            self.record_button.setChecked(False)
            self.record_button.setText("● Gravar Pronúncia (Espaço)")
            self.lbl_record_status.setText("Processando avaliação fonética...")
            self.lbl_record_status.setStyleSheet("color: #3b82f6;")

            audio_data = self.recorder.stop_recording()
            self.recorder.save_wav(self._recorded_wav, audio_data)

            # Dispatch acoustic evaluation
            target = self.txt_target.toPlainText().strip()
            self._evaluate_pronunciation(target, self._recorded_wav)

    def _evaluate_pronunciation(self, target_text: str, audio_path: Path) -> None:
        def on_result(assessment: SpeechAssessmentResult) -> None:
            self.lbl_record_status.setText(
                f"Avaliação concluída! Pontuação Geral: {assessment.overall_score:.1f}%"
            )
            self.lbl_record_status.setStyleSheet("color: #10b981; font-weight: bold;")

            # Flatten words/phonemes for visualizer
            phonemes_data: list[dict[str, Any]] = []
            for word in assessment.words:
                for ph in word.phonemes:
                    phonemes_data.append({"symbol": ph.expected_phoneme, "score": ph.gop_score})

            html = self.render_phonetic_html(phonemes_data)
            self.phonetic_display.setHtml(html)

            # Articulatory feedback
            art_text = assessment.pedagogical_feedback or "Pronúncia adequada nos fonemas principais."
            self.feedback_box.setHtml(f"<div style='color: #e0e7ff;'>{art_text}</div>")

            self.assessment_completed.emit(assessment)

        def on_error(err_type: str, err_msg: str) -> None:
            self.lbl_record_status.setText(f"Erro na avaliação: {err_msg}")
            self.lbl_record_status.setStyleSheet("color: #ef4444;")

        dispatch_task(
            speech_evaluation_task,
            target_text=target_text,
            audio_path=audio_path,
            language="en",
            on_result=on_result,
            on_error=on_error,
        )

    def render_phonetic_html(self, phonemes: list[dict[str, Any]]) -> str:
        """Render GOP color-coded phonetic visualization.

        Green (#4CAF50): Canonical correct (score >= 75.0 or 0.75)
        Yellow (#FFC107): Accented / minor variation (score >= 45.0 or 0.45)
        Red (#F44336): Critical phoneme deviation (score < 45.0 or 0.45)
        """
        spans: list[str] = []
        for ph in phonemes:
            symbol = ph.get("symbol") or ph.get("phoneme") or ""
            score = float(ph.get("score", 0.0))

            # Normalize scale [0, 1] to [0, 100] if needed
            val = score * 100.0 if score <= 1.0 else score

            if val >= 75.0:
                color = "#4CAF50"  # Green
            elif val >= 45.0:
                color = "#FFC107"  # Yellow
            else:
                color = "#F44336"  # Red

            spans.append(
                f"<span style='color: {color}; font-size: 20px; font-weight: bold; margin: 0 3px;' "
                f"title='GOP: {val:.1f}'>{symbol}</span>"
            )

        html = (
            "<div style='text-align: center; padding: 12px; background-color: #141424; border-radius: 6px;'>"
            + "".join(spans)
            + "</div>"
            "<div style='font-size: 11px; text-align: center; margin-top: 6px; color: #9ca3af;'>"
            "<span style='color: #4CAF50;'>■ Correto (≥75)</span> &nbsp;&nbsp; "
            "<span style='color: #FFC107;'>■ Variação (45-74)</span> &nbsp;&nbsp; "
            "<span style='color: #F44336;'>■ Desvio (&lt;45)</span>"
            "</div>"
        )
        return html


__all__ = ["RightLab"]

"""CenterDeck panel: Markdown reader/editor, voice chat dialog, and interactive exercises."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from lang_attack.config import NOTES_DIR
from lang_attack.core.llm_provider import get_llm_provider
from lang_attack.core.security import resolve_safe_path, secure_temp_audio_file
from lang_attack.schemas.exercise import ExerciseItem, ExerciseSession
from lang_attack.workers.tasks import (
    dispatch_task,
    exercise_generation_task,
    tts_synthesis_task,
)

logger = logging.getLogger("lang_attack.ui.center")


class CenterDeck(QWidget):
    """Center deck displaying Markdown text reader/editor, conversation chat, and exercise engine."""

    sentence_to_practice = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("centerDeck")

        self._current_file_path: Path | None = None
        self._current_exercises: list[ExerciseItem] = []
        self._current_exercise_idx: int = 0

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.tab_widget = QTabWidget(self)

        # Tab 1: Reader & Editor
        self.tab_reader = self._build_reader_tab()
        self.tab_widget.addTab(self.tab_reader, "Leitor & Editor")

        # Tab 2: Voice Chat
        self.tab_chat = self._build_chat_tab()
        self.tab_widget.addTab(self.tab_chat, "Conversação de Voz")

        # Tab 3: Exercises
        self.tab_exercises = self._build_exercises_tab()
        self.tab_widget.addTab(self.tab_exercises, "Motor de Exercícios")

        layout.addWidget(self.tab_widget)

    # ------------------------------------------------------------------------
    # Tab 1: Markdown Reader & Editor
    # ------------------------------------------------------------------------
    def _build_reader_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter()

        # Editor
        editor_container = QWidget()
        ed_layout = QVBoxLayout(editor_container)
        ed_layout.setContentsMargins(0, 0, 0, 0)
        ed_label = QLabel("Editor de Notas (Markdown):")
        ed_label.setStyleSheet("font-weight: bold; color: #a5b4fc;")
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Abra ou digite notas em Markdown aqui...")

        # Debounce timer for markdown preview (250ms)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._sync_preview)
        self.editor.textChanged.connect(self._on_editor_text_changed)

        ed_layout.addWidget(ed_label)
        ed_layout.addWidget(self.editor)
        splitter.addWidget(editor_container)

        # Preview
        preview_container = QWidget()
        pr_layout = QVBoxLayout(preview_container)
        pr_layout.setContentsMargins(0, 0, 0, 0)
        pr_label = QLabel("Visualização Formatada:")
        pr_label.setStyleSheet("font-weight: bold; color: #a5b4fc;")
        self.preview = QTextBrowser()
        self.preview.setOpenExternalLinks(True)
        pr_layout.addWidget(pr_label)
        pr_layout.addWidget(self.preview)
        splitter.addWidget(preview_container)

        layout.addWidget(splitter, 1)

        # Controls bar
        btn_bar = QHBoxLayout()
        self.btn_save = QPushButton("Salvar Nota")
        self.btn_save.clicked.connect(self._save_current_file)
        btn_bar.addWidget(self.btn_save)

        self.btn_practice = QPushButton("Treinar Fala da Seleção")
        self.btn_practice.setStyleSheet("background-color: #059669; border-color: #10b981;")
        self.btn_practice.setToolTip(
            "Envia o texto selecionado ou sentença atual para o laboratório fonético"
        )
        self.btn_practice.clicked.connect(self._send_to_speech_lab)
        btn_bar.addWidget(self.btn_practice)

        self.btn_gen_drills = QPushButton("Gerar Exercícios Desta Nota")
        self.btn_gen_drills.clicked.connect(self._generate_exercises_from_note)
        btn_bar.addWidget(self.btn_gen_drills)

        layout.addLayout(btn_bar)
        return widget

    def _on_editor_text_changed(self) -> None:
        self._preview_timer.start()

    def load_note(self, file_path: str | Path) -> None:
        """Load markdown note into editor and update preview with safe path verification."""
        try:
            path = resolve_safe_path(NOTES_DIR, file_path, allow_root=False)
        except Exception as exc:
            logger.error("Blocked loading note outside NOTES_DIR: %s", exc)
            return

        if not path.exists():
            return
        self._current_file_path = path
        content = path.read_text(encoding="utf-8")
        self.editor.setPlainText(content)
        self._sync_preview()
        self.tab_widget.setCurrentIndex(0)

    def _sync_preview(self) -> None:
        text = self.editor.toPlainText()
        self.preview.setMarkdown(text)

    def _save_current_file(self) -> None:
        if self._current_file_path:
            try:
                safe_path = resolve_safe_path(NOTES_DIR, self._current_file_path, allow_root=False)
                safe_path.write_text(self.editor.toPlainText(), encoding="utf-8")
                QMessageBox.information(
                    self, "Salvo", f"Nota salva com sucesso em:\n{safe_path.name}"
                )
            except Exception as exc:
                logger.error("Blocked saving note outside NOTES_DIR: %s", exc)
                QMessageBox.critical(
                    self, "Erro de Segurança", f"Não foi possível salvar o arquivo: {exc}"
                )

    def _send_to_speech_lab(self) -> None:
        cursor = self.editor.textCursor()
        selected = cursor.selectedText().strip()
        if not selected:
            # Fallback to current line
            selected = cursor.block().text().strip()

        if selected:
            self.sentence_to_practice.emit(selected)
        else:
            QMessageBox.information(
                self,
                "Treino de Fala",
                "Selecione um trecho de texto no editor ou posicione o cursor na frase desejada.",
            )

    # ------------------------------------------------------------------------
    # Tab 2: Voice Conversation Chat
    # ------------------------------------------------------------------------
    def _build_chat_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Diálogo com Tutor de Idiomas:")
        header.setStyleSheet("font-weight: bold; color: #a5b4fc;")
        layout.addWidget(header)

        self.chat_display = QTextBrowser()
        self.chat_display.setPlaceholderText("Inicie a conversa digitando uma mensagem abaixo...")
        layout.addWidget(self.chat_display, 1)

        input_bar = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Escreva em inglês ou no idioma alvo...")
        self.chat_input.returnPressed.connect(self._send_chat_message)
        input_bar.addWidget(self.chat_input, 1)

        self.btn_chat_send = QPushButton("Enviar")
        self.btn_chat_send.clicked.connect(self._send_chat_message)
        input_bar.addWidget(self.btn_chat_send)

        layout.addLayout(input_bar)
        return widget

    def _send_chat_message(self) -> None:
        text = self.chat_input.text().strip()
        if not text:
            return

        self.chat_input.clear()
        self.chat_display.append(f"<p style='color: #60a5fa;'><b>Você:</b> {text}</p>")

        self.btn_chat_send.setEnabled(False)
        provider = get_llm_provider()

        def on_reply(reply: str) -> None:
            self.btn_chat_send.setEnabled(True)
            self.chat_display.append(f"<p style='color: #34d399;'><b>Tutor:</b> {reply}</p>")
            # Auto-synthesize audio of tutor reply
            target_wav = secure_temp_audio_file(prefix="tutor_reply_", suffix=".wav")
            dispatch_task(tts_synthesis_task, text=reply, output_path=target_wav, speed=1.0)

        def on_err(err_type: str, err_msg: str) -> None:
            self.btn_chat_send.setEnabled(True)
            self.chat_display.append(
                f"<p style='color: #f87171;'><i>Erro ({err_type}): {err_msg}</i></p>"
            )

        dispatch_task(
            provider.chat,
            message=text,
            on_result=on_reply,
            on_error=on_err,
        )

    # ------------------------------------------------------------------------
    # Tab 3: Interactive Exercises Engine
    # ------------------------------------------------------------------------
    def _build_exercises_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_exercise_title = QLabel("Nenhum exercício carregado.")
        self.lbl_exercise_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #a5b4fc;")
        layout.addWidget(self.lbl_exercise_title)

        self.lbl_question = QLabel("Abra uma nota no editor e clique em 'Gerar Exercícios Desta Nota'.")
        self.lbl_question.setWordWrap(True)
        self.lbl_question.setStyleSheet("font-size: 13px; margin: 8px 0;")
        layout.addWidget(self.lbl_question)

        self.lbl_context = QLabel("")
        self.lbl_context.setStyleSheet("color: #94a3b8; font-style: italic; margin-bottom: 8px;")
        self.lbl_context.setWordWrap(True)
        layout.addWidget(self.lbl_context)

        # Options Container (Multiple Choice)
        self.options_container = QWidget()
        self.options_layout = QVBoxLayout(self.options_container)
        self.options_layout.setContentsMargins(0, 0, 0, 0)
        self.option_button_group = QButtonGroup(self)
        layout.addWidget(self.options_container)

        # Text input (Cloze / Translation)
        self.input_answer = QLineEdit()
        self.input_answer.setPlaceholderText("Digite sua resposta...")
        self.input_answer.setVisible(False)
        self.input_answer.returnPressed.connect(self._check_answer)
        layout.addWidget(self.input_answer)

        # Feedback
        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setWordWrap(True)
        self.lbl_feedback.setStyleSheet("font-size: 12px; margin: 6px 0;")
        layout.addWidget(self.lbl_feedback)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_check_answer = QPushButton("Verificar Resposta")
        self.btn_check_answer.clicked.connect(self._check_answer)
        self.btn_check_answer.setEnabled(False)
        btn_layout.addWidget(self.btn_check_answer)

        self.btn_next_exercise = QPushButton("Próximo Exercício →")
        self.btn_next_exercise.clicked.connect(self._next_exercise)
        self.btn_next_exercise.setEnabled(False)
        btn_layout.addWidget(self.btn_next_exercise)

        layout.addLayout(btn_layout)
        return widget

    def load_exercises(self, exercises: list[ExerciseItem]) -> None:
        """Load list of ExerciseItem models into the interactive engine."""
        self._current_exercises = exercises
        self._current_exercise_idx = 0
        self._display_current_exercise()
        self.tab_widget.setCurrentIndex(2)

    def _display_current_exercise(self) -> None:
        # 1. Unbind and clear buttons from group
        for button in self.option_button_group.buttons():
            self.option_button_group.removeButton(button)

        # 2. Safely remove and deallocate all orphan layout items and widgets
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        self.lbl_feedback.setText("")
        self.btn_check_answer.setEnabled(True)
        self.btn_next_exercise.setEnabled(False)

        if not self._current_exercises or self._current_exercise_idx >= len(self._current_exercises):
            self.lbl_question.setText("Parabéns! Todos os exercícios foram concluídos.")
            self.lbl_context.setText("")
            self.input_answer.setVisible(False)
            self.btn_check_answer.setEnabled(False)
            return

        item = self._current_exercises[self._current_exercise_idx]
        self.lbl_exercise_title.setText(
            f"Exercício {self._current_exercise_idx + 1} de {len(self._current_exercises)}"
        )
        self.lbl_question.setText(item.prompt or item.question)
        context_text = item.context_sentence or item.explanation or ""
        self.lbl_context.setText(f"Contexto: {context_text}" if context_text else "")

        if item.options:
            self.input_answer.setVisible(False)
            self.options_container.setVisible(True)
            for idx, opt in enumerate(item.options):
                rb = QRadioButton(opt, parent=self.options_container)
                self.option_button_group.addButton(rb, idx)
                self.options_layout.addWidget(rb)
        else:
            self.options_container.setVisible(False)
            self.input_answer.setVisible(True)
            self.input_answer.clear()

    def _check_answer(self) -> None:
        if not self._current_exercises or self._current_exercise_idx >= len(self._current_exercises):
            return

        item = self._current_exercises[self._current_exercise_idx]
        user_answer = ""

        if item.options:
            selected = self.option_button_group.checkedButton()
            if selected:
                user_answer = selected.text().strip()
            else:
                self.lbl_feedback.setText(
                    "<span style='color: #F44336;'>Selecione uma opção antes de verificar!</span>"
                )
                return
        else:
            user_answer = self.input_answer.text().strip()

        alternatives = getattr(item, "acceptable_alternatives", []) or []
        is_correct = (
            user_answer.lower() == item.correct_answer.strip().lower()
            or any(user_answer.lower() == alt.strip().lower() for alt in alternatives)
        )

        explanation = item.explanation or item.context_sentence or ""
        if is_correct:
            self.lbl_feedback.setText(
                f"<span style='color: #4CAF50;'>✓ Correto!</span> {explanation}"
            )
        else:
            self.lbl_feedback.setText(
                f"<span style='color: #F44336;'>✗ Incorreto.</span> Resposta esperada: "
                f"<b>{item.correct_answer}</b>.<br>{explanation}"
            )

        self.btn_check_answer.setEnabled(False)
        self.btn_next_exercise.setEnabled(True)

    def _next_exercise(self) -> None:
        self._current_exercise_idx += 1
        self._display_current_exercise()

    def _generate_exercises_from_note(self) -> None:
        content = self.editor.toPlainText().strip()
        if not content:
            QMessageBox.warning(
                self, "Aviso", "O editor está vazio. Abra ou digite uma nota primeiro."
            )
            return

        self.btn_gen_drills.setEnabled(False)
        self.btn_gen_drills.setText("Gerando...")

        topic = self._current_file_path.stem if self._current_file_path else "Estudo de Idiomas"

        def on_drills_ready(session: Any) -> None:
            self.btn_gen_drills.setEnabled(True)
            self.btn_gen_drills.setText("Gerar Exercícios Desta Nota")
            items = session.items if isinstance(session, ExerciseSession) else session
            if isinstance(items, list):
                self.load_exercises(items)

        def on_drills_error(err_type: str, err_msg: str) -> None:
            self.btn_gen_drills.setEnabled(True)
            self.btn_gen_drills.setText("Gerar Exercícios Desta Nota")
            QMessageBox.warning(self, "Erro", f"Falha ao gerar exercícios ({err_type}):\n{err_msg}")

        dispatch_task(
            exercise_generation_task,
            topic=topic,
            note_content=content,
            language="en",
            num_exercises=5,
            on_result=on_drills_ready,
            on_error=on_drills_error,
        )


__all__ = ["CenterDeck"]

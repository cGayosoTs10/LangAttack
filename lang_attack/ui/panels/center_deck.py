"""CenterDeck panel: Markdown reader/editor, voice chat dialog, and interactive exercises."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Signal
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

from lang_attack.core.llm_provider import get_llm_provider
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
        self.editor.textChanged.connect(self._sync_preview)
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
        self.btn_practice.setToolTip("Envia o texto selecionado ou sentença atual para o laboratório fonético")
        self.btn_practice.clicked.connect(self._send_to_speech_lab)
        btn_bar.addWidget(self.btn_practice)

        self.btn_gen_drills = QPushButton("Gerar Exercícios Desta Nota")
        self.btn_gen_drills.clicked.connect(self._generate_exercises_from_note)
        btn_bar.addWidget(self.btn_gen_drills)

        layout.addLayout(btn_bar)
        return widget

    def load_note(self, file_path: str | Path) -> None:
        """Load markdown note into editor and update preview."""
        path = Path(file_path)
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
            self._current_file_path.write_text(self.editor.toPlainText(), encoding="utf-8")
            QMessageBox.information(self, "Salvo", f"Nota salva com sucesso em:\n{self._current_file_path.name}")

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
                "Selecione uma frase ou palavra no texto para praticar no laboratório fonético.",
            )

    # ------------------------------------------------------------------------
    # Tab 2: Voice Chat
    # ------------------------------------------------------------------------
    def _build_chat_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.chat_display = QTextBrowser()
        self.chat_display.append(
            "<div style='color: #818cf8; font-weight: bold;'>Tutor de Idiomas:</div>"
            "<div style='background-color: #1e1b4b; padding: 8px; border-radius: 6px; margin-bottom: 8px;'>"
            "Olá! Escolha um idioma e envie qualquer mensagem ou pergunta para conversarmos. "
            "Respondo com texto e áudio sintetizado!</div>"
        )
        layout.addWidget(self.chat_display, 1)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Digite sua mensagem para o tutor...")
        self.chat_input.returnPressed.connect(self._send_chat_message)
        input_layout.addWidget(self.chat_input, 1)

        self.btn_send_chat = QPushButton("Enviar")
        self.btn_send_chat.clicked.connect(self._send_chat_message)
        input_layout.addWidget(self.btn_send_chat, 0)

        layout.addLayout(input_layout)
        return widget

    def _send_chat_message(self) -> None:
        user_text = self.chat_input.text().strip()
        if not user_text:
            return

        self.chat_input.clear()
        self.chat_display.append(
            f"<div style='color: #34d399; font-weight: bold; text-align: right;'>Você:</div>"
            f"<div style='background-color: #064e3b; padding: 8px; border-radius: 6px; margin-bottom: 8px; text-align: right;'>"
            f"{user_text}</div>"
        )

        self.btn_send_chat.setEnabled(False)
        self.btn_send_chat.setText("Pensando...")

        def llm_worker_fn() -> str:
            provider = get_llm_provider()
            prompt = (
                f"You are a supportive language tutor. Reply concisely and naturally to the student in the target language.\n"
                f"Student: {user_text}\nTutor:"
            )
            return provider.generate(prompt)

        def on_llm_result(tutor_reply: str) -> None:
            self.btn_send_chat.setEnabled(True)
            self.btn_send_chat.setText("Enviar")
            self.chat_display.append(
                f"<div style='color: #818cf8; font-weight: bold;'>Tutor:</div>"
                f"<div style='background-color: #1e1b4b; padding: 8px; border-radius: 6px; margin-bottom: 8px;'>"
                f"{tutor_reply}</div>"
            )
            # Synthesize voice audio
            temp_wav = Path("/tmp/tutor_reply.wav")
            dispatch_task(tts_synthesis_task, text=tutor_reply, output_path=temp_wav)

        def on_llm_error(err_type: str, err_msg: str) -> None:
            self.btn_send_chat.setEnabled(True)
            self.btn_send_chat.setText("Enviar")
            self.chat_display.append(
                f"<div style='color: #f87171;'>Erro na resposta do tutor ({err_type}): {err_msg}</div>"
            )

        dispatch_task(llm_worker_fn, on_result=on_llm_result, on_error=on_llm_error)

    # ------------------------------------------------------------------------
    # Tab 3: Exercise Engine
    # ------------------------------------------------------------------------
    def _build_exercises_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.lbl_exercise_title = QLabel("Exercícios de Fixação:")
        self.lbl_exercise_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #a5b4fc;")
        layout.addWidget(self.lbl_exercise_title)

        self.lbl_question = QLabel("Nenhum exercício carregado no momento.")
        self.lbl_question.setWordWrap(True)
        self.lbl_question.setStyleSheet("font-size: 14px; color: #ffffff; padding: 8px; background: #1c1c30; border-radius: 6px;")
        layout.addWidget(self.lbl_question)

        self.lbl_context = QLabel("")
        self.lbl_context.setStyleSheet("font-style: italic; color: #9ca3af; padding-left: 8px;")
        self.lbl_context.setWordWrap(True)
        layout.addWidget(self.lbl_context)

        # Options Container
        self.options_container = QWidget()
        self.options_layout = QVBoxLayout(self.options_container)
        self.option_button_group = QButtonGroup(self)
        layout.addWidget(self.options_container)

        # Fill blank input
        self.input_answer = QLineEdit()
        self.input_answer.setPlaceholderText("Digite sua resposta...")
        self.input_answer.setVisible(False)
        layout.addWidget(self.input_answer)

        # Feedback
        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setWordWrap(True)
        self.lbl_feedback.setStyleSheet("font-weight: bold; font-size: 13px; padding: 6px;")
        layout.addWidget(self.lbl_feedback)

        layout.addStretch(1)

        # Exercise Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_check_answer = QPushButton("Verificar Resposta")
        self.btn_check_answer.setStyleSheet("background-color: #4f46e5;")
        self.btn_check_answer.clicked.connect(self._check_answer)
        btn_layout.addWidget(self.btn_check_answer)

        self.btn_next_exercise = QPushButton("Próximo Exercício")
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
        # Clear options
        for button in self.option_button_group.buttons():
            self.option_button_group.removeButton(button)
            button.deleteLater()

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
        self.lbl_exercise_title.setText(f"Exercício {self._current_exercise_idx + 1} de {len(self._current_exercises)}")
        self.lbl_question.setText(item.question)
        self.lbl_context.setText(f"Contexto: {item.context_sentence}" if item.context_sentence else "")

        if item.options:
            self.input_answer.setVisible(False)
            self.options_container.setVisible(True)
            for idx, opt in enumerate(item.options):
                rb = QRadioButton(opt)
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
                self.lbl_feedback.setText("<span style='color: #F44336;'>Selecione uma opção antes de verificar!</span>")
                return
        else:
            user_answer = self.input_answer.text().strip()

        is_correct = (
            user_answer.lower() == item.correct_answer.strip().lower()
            or any(user_answer.lower() == alt.strip().lower() for alt in item.acceptable_alternatives)
        )

        if is_correct:
            self.lbl_feedback.setText(
                f"<span style='color: #4CAF50;'>✓ Correto!</span> {item.explanation}"
            )
        else:
            self.lbl_feedback.setText(
                f"<span style='color: #F44336;'>✗ Incorreto.</span> Resposta esperada: "
                f"<b>{item.correct_answer}</b>.<br>{item.explanation}"
            )

        self.btn_check_answer.setEnabled(False)
        self.btn_next_exercise.setEnabled(True)

    def _next_exercise(self) -> None:
        self._current_exercise_idx += 1
        self._display_current_exercise()

    def _generate_exercises_from_note(self) -> None:
        content = self.editor.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "Aviso", "O editor está vazio. Abra ou digite uma nota primeiro.")
            return

        self.btn_gen_drills.setEnabled(False)
        self.btn_gen_drills.setText("Gerando...")

        topic = self._current_file_path.stem if self._current_file_path else "Estudo de Idiomas"

        def on_drills_ready(session: ExerciseSession) -> None:
            self.btn_gen_drills.setEnabled(True)
            self.btn_gen_drills.setText("Gerar Exercícios Desta Nota")
            self.load_exercises(session.items)

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

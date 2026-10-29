"""LeftWorkspace panel: Dynamic notes vault navigation, study context selector, and URL importer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileSystemModel,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from lang_attack.config import NOTES_DIR
from lang_attack.core.security import resolve_safe_path, sanitize_filename
from lang_attack.workers.tasks import dispatch_task, ingest_url_task

logger = logging.getLogger("lang_attack.ui.left")


class LeftWorkspace(QWidget):
    """File explorer workspace for managing markdown notes, study contexts, and content imports."""

    note_selected = Signal(str)
    context_changed = Signal(str)
    file_imported = Signal(str)

    def __init__(self, notes_dir: Path | str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("leftWorkspace")

        # Resolve and ensure notes directory
        if notes_dir is None:
            self.notes_dir = Path(NOTES_DIR).resolve()
        else:
            self.notes_dir = Path(notes_dir).resolve()
        self.notes_dir.mkdir(parents=True, exist_ok=True)

        self._init_ui()
        self._populate_contexts()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 1. Top Section: Study Context Selector
        context_header = QLabel("Espaço de Notas & Idioma:")
        context_header.setStyleSheet("font-weight: bold; color: #a5b4fc;")
        layout.addWidget(context_header)

        context_layout = QHBoxLayout()
        self.context_combo = QComboBox()
        self.context_combo.setToolTip("Selecione o idioma ou pasta de contexto")
        self.context_combo.currentTextChanged.connect(self._on_context_combo_changed)

        self.add_folder_btn = QPushButton("+")
        self.add_folder_btn.setFixedWidth(32)
        self.add_folder_btn.setToolTip("Criar nova pasta de idioma na raiz")
        self.add_folder_btn.clicked.connect(self._create_root_folder)

        context_layout.addWidget(self.context_combo, 1)
        context_layout.addWidget(self.add_folder_btn, 0)
        layout.addLayout(context_layout)

        # 2. Middle Section: Notes File Tree
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(str(self.notes_dir))
        self.file_model.setNameFilters(["*.md", "*.markdown"])
        self.file_model.setNameFilterDisables(False)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(str(self.notes_dir)))
        self.tree_view.setAnimated(True)
        self.tree_view.setIndentation(16)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._open_context_menu)
        self.tree_view.doubleClicked.connect(self._on_item_double_clicked)

        # Hide extra size/type/date columns, keep name
        header = self.tree_view.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 4):
            self.tree_view.setColumnHidden(col, True)

        layout.addWidget(self.tree_view, 1)

        # 3. Bottom Section: URL Ingestion
        import_header = QLabel("Importar Conteúdo Web / Vídeo:")
        import_header.setStyleSheet("font-weight: bold; color: #a5b4fc; margin-top: 4px;")
        layout.addWidget(import_header)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Cole URL de artigo web ou YouTube...")
        layout.addWidget(self.url_input)

        self.import_btn = QPushButton("Importar")
        self.import_btn.clicked.connect(self._import_url)
        layout.addWidget(self.import_btn)

    def _populate_contexts(self) -> None:
        """Scan top-level subfolders of notes_dir and populate context_combo."""
        current_text = self.context_combo.currentText()
        self.context_combo.blockSignals(True)
        self.context_combo.clear()
        self.context_combo.addItem("Todos (Raiz)", str(self.notes_dir))

        if self.notes_dir.exists():
            for item in sorted(self.notes_dir.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    self.context_combo.addItem(item.name, str(item))

        # Restore selection if possible
        idx = self.context_combo.findText(current_text)
        if idx >= 0:
            self.context_combo.setCurrentIndex(idx)
        else:
            self.context_combo.setCurrentIndex(0)
        self.context_combo.blockSignals(False)

    def _on_context_combo_changed(self, text: str) -> None:
        path_str = self.context_combo.currentData()
        if path_str and Path(path_str).exists():
            self.tree_view.setRootIndex(self.file_model.index(path_str))
        else:
            self.tree_view.setRootIndex(self.file_model.index(str(self.notes_dir)))
        self.context_changed.emit(text)

    def _create_root_folder(self) -> None:
        folder_name, ok = QInputDialog.getText(
            self,
            "Nova Pasta de Idioma",
            "Nome da nova pasta/idioma:",
        )
        if ok and folder_name.strip():
            try:
                clean_name = sanitize_filename(folder_name.strip(), default_name="novo_idioma")
                new_path = resolve_safe_path(self.notes_dir, clean_name, allow_root=False)
                new_path.mkdir(parents=True, exist_ok=True)
                self._populate_contexts()
                idx = self.context_combo.findText(clean_name)
                if idx >= 0:
                    self.context_combo.setCurrentIndex(idx)
            except Exception as exc:
                logger.warning("Falha ao criar pasta de idioma: %s", exc)
                QMessageBox.warning(self, "Nome Inválido", f"Não foi possível criar a pasta: {exc}")

    def _open_context_menu(self, point: Any) -> None:
        index = self.tree_view.indexAt(point)
        target_dir = self.notes_dir

        if index.isValid():
            path = Path(self.file_model.filePath(index))
            target_dir = path if path.is_dir() else path.parent

        menu = QMenu(self)

        action_new_folder = menu.addAction("Nova Pasta")
        action_new_file = menu.addAction("Novo Arquivo .md")
        menu.addSeparator()
        action_rename = menu.addAction("Renomear")
        action_delete = menu.addAction("Excluir")

        if not index.isValid():
            action_rename.setEnabled(False)
            action_delete.setEnabled(False)

        chosen = menu.exec(self.tree_view.viewport().mapToGlobal(point))

        if chosen == action_new_folder:
            name, ok = QInputDialog.getText(self, "Nova Pasta", "Nome da subpasta:")
            if ok and name.strip():
                try:
                    clean_name = sanitize_filename(name.strip(), default_name="nova_pasta")
                    new_path = resolve_safe_path(target_dir, clean_name, allow_root=False)
                    resolve_safe_path(self.notes_dir, new_path, allow_root=False)
                    new_path.mkdir(parents=True, exist_ok=True)
                    self._populate_contexts()
                except Exception as exc:
                    logger.warning("Falha ao criar subpasta: %s", exc)
                    QMessageBox.warning(self, "Nome Inválido", f"Nome de pasta inválido: {exc}")

        elif chosen == action_new_file:
            name, ok = QInputDialog.getText(self, "Novo Arquivo", "Nome do arquivo Markdown:")
            if ok and name.strip():
                try:
                    clean_name = sanitize_filename(name.strip(), default_name="nota")
                    if not clean_name.lower().endswith(".md"):
                        clean_name += ".md"
                    new_file = resolve_safe_path(target_dir, clean_name, allow_root=False)
                    resolve_safe_path(self.notes_dir, new_file, allow_root=False)
                    if not new_file.exists():
                        new_file.write_text(f"# {clean_name[:-3]}\n\n", encoding="utf-8")
                    self.note_selected.emit(str(new_file))
                except Exception as exc:
                    logger.warning("Falha ao criar arquivo de nota: %s", exc)
                    QMessageBox.warning(self, "Nome Inválido", f"Nome de arquivo inválido: {exc}")

        elif chosen == action_rename and index.isValid():
            self.tree_view.edit(index)

        elif chosen == action_delete and index.isValid():
            path = Path(self.file_model.filePath(index))
            try:
                resolved_path = resolve_safe_path(self.notes_dir, path, allow_root=False)
            except ValueError as exc:
                logger.error("Tentativa de exclusão fora do diretório de notas: %s", exc)
                QMessageBox.critical(
                    self,
                    "Acesso Negado",
                    "Operação de exclusão inválida ou fora do cofre de notas.",
                )
                return

            reply = QMessageBox.question(
                self,
                "Confirmar Exclusão",
                f"Tem certeza que deseja excluir '{resolved_path.name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if resolved_path.is_dir():
                    self.file_model.remove(index)
                    self._populate_contexts()
                else:
                    self.file_model.remove(index)

    def _on_item_double_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        path = Path(self.file_model.filePath(index))
        try:
            resolved = resolve_safe_path(self.notes_dir, path, allow_root=False)
            if resolved.is_file() and resolved.suffix.lower() in (".md", ".markdown"):
                self.note_selected.emit(str(resolved))
        except Exception as exc:
            logger.warning("Item selecionado fora do diretório seguro: %s", exc)

    def _import_url(self) -> None:
        url = self.url_input.text().strip()
        if not url:
            return

        self.import_btn.setEnabled(False)
        self.import_btn.setText("Importando...")

        # Target active folder
        active_path = self.context_combo.currentData() or str(self.notes_dir)
        try:
            target_dir = resolve_safe_path(self.notes_dir, active_path, allow_root=True)
        except Exception:
            target_dir = self.notes_dir

        if not target_dir.exists():
            target_dir = self.notes_dir

        def on_success(result: dict[str, Any]) -> None:
            self.import_btn.setEnabled(True)
            self.import_btn.setText("Importar")
            self.url_input.clear()
            saved_path = result.get("path")
            if saved_path:
                self.file_imported.emit(str(saved_path))
                self.note_selected.emit(str(saved_path))
            QMessageBox.information(
                self,
                "Importação Concluída",
                f"Conteúdo importado com sucesso:\n{result.get('title', 'Sem título')}",
            )

        def on_error(err_type: str, err_msg: str) -> None:
            self.import_btn.setEnabled(True)
            self.import_btn.setText("Importar")
            QMessageBox.warning(
                self,
                "Erro na Importação",
                f"Falha ao importar URL ({err_type}):\n{err_msg}",
            )

        dispatch_task(
            ingest_url_task,
            url=url,
            output_dir=target_dir,
            on_result=on_success,
            on_error=on_error,
        )


__all__ = ["LeftWorkspace"]

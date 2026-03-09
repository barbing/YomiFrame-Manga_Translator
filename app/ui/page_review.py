# -*- coding: utf-8 -*-
"""Split view page review dialog."""
from __future__ import annotations
import os
from typing import Any, Dict
from PySide6 import QtCore, QtGui, QtWidgets
from app.io.project import default_project_dict, load_project, save_project
from app.render.renderer import render_translations


class ResizableLabel(QtWidgets.QLabel):
    """QLabel that scales its pixmap to fill the available space."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._pixmap = None

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        super().setPixmap(self._scaled_pixmap())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self._pixmap:
            super().setPixmap(self._scaled_pixmap())

    def _scaled_pixmap(self) -> QtGui.QPixmap:
        if not self._pixmap or self._pixmap.isNull():
            return self._pixmap
        return self._pixmap.scaled(
            self.size(), 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )

class PageReviewDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        page_record: Dict[str, Any] | None = None,
        json_path: str = "",
        output_suffix: str = "_translated",
        font_name: str = "",
        inpaint_mode: str = "fast",
        use_gpu: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Page")
        self.resize(1320, 880)
        self.setMinimumSize(1200, 820)
        self._page = page_record or {}
        self._json_path = json_path
        self._output_suffix = output_suffix or "_translated"
        self._font_name = font_name
        self._inpaint_mode = inpaint_mode
        self._use_gpu = use_gpu
        self._setup_ui()
        self._load_page()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        layout.addWidget(split, 10)

        left = QtWidgets.QWidget(split)
        left_layout = QtWidgets.QVBoxLayout(left)
        self.original_label = QtWidgets.QLabel("Original")
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_view = ResizableLabel()
        left_layout.addWidget(self.original_label)
        left_layout.addWidget(self.original_view, 1)

        right = QtWidgets.QWidget(split)
        right_layout = QtWidgets.QVBoxLayout(right)
        self.translated_label = QtWidgets.QLabel("Translated")
        self.translated_label.setAlignment(QtCore.Qt.AlignCenter)
        self.translated_view = ResizableLabel()
        right_layout.addWidget(self.translated_label)
        right_layout.addWidget(self.translated_view, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Region", "OCR", "Translation", "Needs Review"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 2)

        footer = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton("Save to JSON")
        self.render_btn = QtWidgets.QPushButton("Re-render Page")
        self.close_btn = QtWidgets.QPushButton("Close")
        footer.addWidget(self.save_btn)
        footer.addWidget(self.render_btn)
        footer.addStretch(1)
        footer.addWidget(self.close_btn)
        layout.addLayout(footer)

        self.save_btn.clicked.connect(self._save_json)
        self.render_btn.clicked.connect(self._rerender)
        self.close_btn.clicked.connect(self.accept)

    def _load_page(self) -> None:
        image_path = self._page.get("image_path", "")
        output_path = self._page.get("output_path", "")
        if image_path and os.path.isfile(image_path):
            pixmap = QtGui.QPixmap(image_path)
            self.original_view.setPixmap(pixmap)
        if output_path and os.path.isfile(output_path):
            pixmap = QtGui.QPixmap(output_path)
            self.translated_view.setPixmap(pixmap)
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for region in self._page.get("regions", []):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(region.get("region_id", ""))))
            ocr_item = QtWidgets.QTableWidgetItem(str(region.get("ocr_text", "")))
            ocr_item.setFlags(ocr_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 1, ocr_item)
            trans_item = QtWidgets.QTableWidgetItem(str(region.get("translation", "")))
            self.table.setItem(row, 2, trans_item)
            check_item = QtWidgets.QTableWidgetItem("")
            check_item.setFlags(check_item.flags() | QtCore.Qt.ItemIsUserCheckable)
            check_item.setCheckState(QtCore.Qt.Checked if region.get("flags", {}).get("needs_review") else QtCore.Qt.Unchecked)
            self.table.setItem(row, 3, check_item)

    def _apply_table(self) -> None:
        regions = self._page.get("regions", [])
        for row in range(self.table.rowCount()):
            if row >= len(regions):
                continue
            region = regions[row]
            region["translation"] = self.table.item(row, 2).text().strip()
            flags = region.get("flags", {})
            flags["needs_review"] = self.table.item(row, 3).checkState() == QtCore.Qt.Checked
            region["flags"] = flags

    def _save_json(self) -> None:
        if not self._json_path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project JSON", filter="JSON Files (*.json)")
            if not path:
                return
            self._json_path = path
        self._apply_table()
        if os.path.exists(self._json_path):
            data = load_project(self._json_path)
        else:
            data = default_project_dict()
        updated = False
        for page in data.get("pages", []):
            if page.get("image_path") == self._page.get("image_path"):
                page["regions"] = self._page.get("regions", [])
                updated = True
                break
        if not updated:
            data.setdefault("pages", []).append(self._page)
        save_project(self._json_path, data)

    def _rerender(self) -> None:
        self._apply_table()
        image_path = self._page.get("image_path", "")
        output_path = self._page.get("output_path") or _default_output_path(image_path, self._output_suffix)
        if not image_path or not output_path:
            return
        render_translations(
            image_path,
            output_path,
            self._page.get("regions", []),
            self._font_name,
            inpaint_mode=self._inpaint_mode,
            use_gpu=self._use_gpu,
        )
        if os.path.isfile(output_path):
            pixmap = QtGui.QPixmap(output_path)
            self.translated_view.setPixmap(pixmap)


def _default_output_path(image_path: str, suffix: str) -> str:
    if not image_path:
        return ""
    folder = os.path.dirname(image_path)
    name, ext = os.path.splitext(os.path.basename(image_path))
    return os.path.join(folder, f"{name}{suffix}{ext}")

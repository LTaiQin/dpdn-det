import sys
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSlider, QTabWidget,
    QMessageBox, QFrame, QGroupBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QUrl
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor


class DetectionThread(QThread):
    """检测线程"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, server_url, image_path, score_threshold):
        super().__init__()
        self.server_url = server_url
        self.image_path = image_path
        self.score_threshold = score_threshold

    def run(self):
        try:
            self.status.emit("正在上传图片...")

            img_bytes = Path(self.image_path).read_bytes()

            self.status.emit("正在发送请求...")

            payload = {
                "image_b64": base64.b64encode(img_bytes).decode("utf-8"),
                "score_thresh": float(self.score_threshold),
            }

            url = self.server_url.rstrip("/") + "/infer"
            data = json.dumps(payload).encode("utf-8")
            req = Request(url, data=data, headers={"Content-Type": "application/json"})

            with urlopen(req, timeout=120) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            if not resp_data.get("ok", False):
                raise Exception(f"服务器错误: {resp_data.get('error')}")

            self.status.emit("正在处理结果...")
            result = resp_data["result"]
            result["_input_image_path"] = self.image_path
            self.finished.emit(result)

        except URLError as e:
            self.error.emit(f"网络错误: 无法连接到服务器\n{str(e)}")
        except HTTPError as e:
            self.error.emit(f"HTTP错误: {e.code} - {e.reason}")
        except Exception as e:
            self.error.emit(f"检测失败: {str(e)}")


class ImageDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Defaults must be set before building UI components that reference them.
        self.image_path = ""
        self.server_url = "http://127.0.0.1:18080"
        self.output_dir = r"D:\pyCharmProjects\server\output"
        self.score_threshold = 0.5
        self.auto_open_output = False
        self._queue = []  # list[str]
        self._busy = False
        self._last_saved_paths = {}  # image_path -> (out_img, out_json)
        self._settings = QSettings("second_14.1", "remote_infer_gui")
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle("AI 图像检测工具")
        self.setGeometry(200, 100, 900, 700)
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)

        # 应用样式
        self.set_stylesheet()

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # 左侧面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # 右侧面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)

    def set_stylesheet(self):
        """设置现代化样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QWidget {
                font-family: 'Segoe UI', Arial;
                font-size: 11pt;
            }
            QGroupBox {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 20px;
                font-weight: bold;
                color: #333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                background-color: #f5f5f7;
            }
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                background-color: white;
                font-size: 11pt;
            }
            QLineEdit:focus {
                border: 2px solid #007AFF;
            }
            QPushButton {
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                background-color: #007AFF;
                color: white;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004494;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 10px;
                background-color: white;
            }
            QTabBar::tab {
                padding: 10px 25px;
                background-color: #e0e0e0;
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #007AFF;
                font-weight: bold;
            }
            QLabel {
                color: #333;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                background: #007AFF;
                border-radius: 9px;
                margin: -6px 0;
            }
        """)

    def create_left_panel(self):
        """创建左侧设置面板"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet("background-color: white; border-radius: 15px;")
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel("⚙️ 设置")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 服务器地址组
        server_group = QGroupBox("服务器配置")
        server_layout = QVBoxLayout()

        server_label = QLabel("服务器地址:")
        self.server_input = QLineEdit(self.server_url)
        server_layout.addWidget(server_label)
        server_layout.addWidget(self.server_input)

        server_btn_row = QHBoxLayout()
        self.health_btn = QPushButton("🩺 测试连接")
        self.health_btn.setFixedWidth(140)
        self.health_btn.clicked.connect(self.check_server_health)
        self.copy_ssh_btn = QPushButton("📋 复制SSH转发命令")
        self.copy_ssh_btn.clicked.connect(self.copy_ssh_command)
        server_btn_row.addWidget(self.health_btn)
        server_btn_row.addWidget(self.copy_ssh_btn)
        server_layout.addLayout(server_btn_row)

        output_label = QLabel("输出目录:")
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit(self.output_dir)
        browse_btn = QPushButton("浏览")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(browse_btn)
        server_layout.addWidget(output_label)
        server_layout.addLayout(output_layout)

        self.auto_open_chk = QCheckBox("检测完成后自动打开输出目录")
        self.auto_open_chk.setChecked(self.auto_open_output)
        self.auto_open_chk.stateChanged.connect(self.on_auto_open_changed)
        server_layout.addWidget(self.auto_open_chk)

        server_group.setLayout(server_layout)
        layout.addWidget(server_group)

        # 置信度阈值组
        threshold_group = QGroupBox("检测参数")
        threshold_layout = QVBoxLayout()

        self.score_label = QLabel(f"置信度阈值: {self.score_threshold:.2f}")
        self.score_slider = QSlider(Qt.Horizontal)
        self.score_slider.setRange(0, 100)
        self.score_slider.setValue(int(self.score_threshold * 100))
        self.score_slider.valueChanged.connect(self.update_score)

        threshold_layout.addWidget(self.score_label)
        threshold_layout.addWidget(self.score_slider)

        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        # 图片选择组
        image_group = QGroupBox("图片选择")
        image_layout = QVBoxLayout()

        self.select_btn = QPushButton("📁 选择图片（单张）")
        self.select_btn.clicked.connect(self.select_image)
        image_layout.addWidget(self.select_btn)

        self.select_multi_btn = QPushButton("🗂️ 选择图片（批量）")
        self.select_multi_btn.clicked.connect(self.select_images_batch)
        image_layout.addWidget(self.select_multi_btn)

        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("color: #888; font-style: italic;")
        self.file_label.setWordWrap(True)
        image_layout.addWidget(self.file_label)

        queue_title = QLabel("队列（可拖拽图片到窗口）:")
        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue_list.setMinimumHeight(120)
        image_layout.addWidget(queue_title)
        image_layout.addWidget(self.queue_list)

        queue_btns = QHBoxLayout()
        self.queue_clear_btn = QPushButton("🧹 清空")
        self.queue_clear_btn.clicked.connect(self.clear_queue)
        self.queue_remove_btn = QPushButton("🗑️ 移除选中")
        self.queue_remove_btn.clicked.connect(self.remove_selected_queue_items)
        queue_btns.addWidget(self.queue_clear_btn)
        queue_btns.addWidget(self.queue_remove_btn)
        image_layout.addLayout(queue_btns)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        image_layout.addWidget(self.progress)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        layout.addStretch()

        # 检测按钮
        self.detect_btn = QPushButton("🚀 开始检测（当前/队列）")
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                font-size: 14pt;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #30B050;
            }
        """)
        self.detect_btn.clicked.connect(self.start_detection)
        layout.addWidget(self.detect_btn)

        return panel

    def create_right_panel(self):
        """创建右侧预览面板"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet("background-color: white; border-radius: 15px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 标签页
        self.tab_widget = QTabWidget()

        # 原始图片标签
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.original_label = QLabel("请选择图片进行预览")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("color: #888; font-size: 14pt;")
        original_layout.addWidget(self.original_label)
        self.tab_widget.addTab(self.original_tab, "📷 原始图片")

        # 检测结果标签
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        self.result_label = QLabel("检测完成后显示结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #888; font-size: 14pt;")
        result_layout.addWidget(self.result_label)
        self.tab_widget.addTab(self.result_tab, "✅ 检测结果")

        # 检测详情（表格）
        self.detail_tab = QWidget()
        detail_layout = QVBoxLayout(self.detail_tab)
        self.detail_title = QLabel("检测详情（类别/分数/框）")
        self.detail_title.setStyleSheet("font-weight: bold;")
        detail_layout.addWidget(self.detail_title)
        self.det_table = QTableWidget(0, 4)
        self.det_table.setHorizontalHeaderLabels(["class_id", "class_name", "score", "bbox_xyxy"])
        self.det_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.det_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.det_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.det_table.setAlternatingRowColors(True)
        detail_layout.addWidget(self.det_table)

        detail_btns = QHBoxLayout()
        self.open_output_btn = QPushButton("📂 打开输出目录")
        self.open_output_btn.clicked.connect(self.open_output_dir)
        self.open_last_btn = QPushButton("🖼️ 打开最近结果图片")
        self.open_last_btn.clicked.connect(self.open_last_result_image)
        detail_btns.addWidget(self.open_output_btn)
        detail_btns.addWidget(self.open_last_btn)
        detail_layout.addLayout(detail_btns)
        self.tab_widget.addTab(self.detail_tab, "📊 检测详情")

        layout.addWidget(self.tab_widget)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 8px;
                border-radius: 6px;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.status_label)

        return panel

    def update_score(self, value):
        """更新分数标签"""
        self.score_threshold = value / 100
        self.score_label.setText(f"置信度阈值: {self.score_threshold:.2f}")

    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_input.setText(dir_path)
            self.save_settings()

    def select_image(self):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)"
        )

        if file_path:
            self.image_path = file_path
            self.file_label.setText(Path(file_path).name)
            self.file_label.setStyleSheet("color: #333; font-weight: bold;")
            self.status_label.setText(f"已选择: {Path(file_path).name}")

            # 显示原始图片
            self.display_image(file_path, self.original_label)
            self.enqueue_images([file_path], set_current=False)
            self.save_settings()

    def select_images_batch(self):
        """批量选择图片"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片（批量）",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)",
        )
        if file_paths:
            if not self.image_path:
                self.image_path = file_paths[0]
                self.file_label.setText(Path(self.image_path).name)
                self.file_label.setStyleSheet("color: #333; font-weight: bold;")
                self.display_image(self.image_path, self.original_label)
            self.enqueue_images(file_paths, set_current=False)
            self.status_label.setText(f"已加入队列: {len(file_paths)} 张")
            self.save_settings()

    def enqueue_images(self, paths, set_current: bool = False):
        added = 0
        for p in paths:
            p = str(Path(p))
            if p not in self._queue:
                self._queue.append(p)
                item = QListWidgetItem(Path(p).name)
                item.setToolTip(p)
                self.queue_list.addItem(item)
                added += 1
        if set_current and paths:
            self.image_path = str(Path(paths[0]))
        if added:
            self.update_progress()

    def clear_queue(self):
        self._queue = []
        self.queue_list.clear()
        self.update_progress()

    def remove_selected_queue_items(self):
        selected = self.queue_list.selectedItems()
        if not selected:
            return
        names_to_remove = set(i.toolTip() for i in selected)
        self._queue = [p for p in self._queue if p not in names_to_remove]
        for i in selected:
            self.queue_list.takeItem(self.queue_list.row(i))
        self.update_progress()

    def update_progress(self, done: int = 0):
        total = max(len(self._queue), 1)
        self.progress.setMaximum(total)
        self.progress.setValue(min(done, total))
        self.progress.setFormat(f"队列进度: {min(done, total)}/{total}")

    def display_image(self, image_path, label):
        """显示图片"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # 缩放图片以适应标签
            scaled_pixmap = pixmap.scaled(
                500, 500,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setText("")

    def display_result_image(self, image_bytes, label):
        """显示结果图片"""
        pixmap = QPixmap()
        if pixmap.loadFromData(image_bytes):
            scaled_pixmap = pixmap.scaled(
                500, 500,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
            label.setText("")

    def start_detection(self):
        """开始检测"""
        if self._busy:
            QMessageBox.information(self, "提示", "当前正在检测，请稍候…")
            return

        if not self.image_path and not self._queue:
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return

        # If current image isn't in queue, enqueue it to unify flow.
        if self.image_path:
            self.enqueue_images([self.image_path], set_current=False)

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("⏳ 检测中...")
        self._busy = True
        self._queue_done = 0
        self.run_next_in_queue()

    def run_next_in_queue(self):
        if not self._queue:
            self._busy = False
            self.detect_btn.setEnabled(True)
            self.detect_btn.setText("🚀 开始检测（当前/队列）")
            self.status_label.setText("队列已完成")
            self.update_progress(done=0)
            return

        image_path = self._queue[0]
        self.image_path = image_path
        self.file_label.setText(Path(image_path).name)
        self.file_label.setStyleSheet("color: #333; font-weight: bold;")
        self.display_image(image_path, self.original_label)
        self.tab_widget.setCurrentIndex(0)

        self.detection_thread = DetectionThread(
            self.server_input.text(),
            image_path,
            self.score_threshold
        )
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.status.connect(self.on_status_update)
        self.detection_thread.start()

    def on_detection_finished(self, result):
        """检测完成"""
        try:
            # 保存结果
            vis_bytes = base64.b64decode(result["vis_jpg_b64"])

            out_dir = Path(self.output_input.text())
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(self.image_path).stem
            out_img = out_dir / f"{stem}_vis.jpg"
            out_json = out_dir / f"{stem}_det.json"

            out_img.write_bytes(vis_bytes)
            out_json.write_text(
                json.dumps(result.get("detections", []), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            self._last_saved_paths[str(Path(self.image_path))] = (str(out_img), str(out_json))

            # 显示结果
            self.display_result_image(vis_bytes, self.result_label)
            self.tab_widget.setCurrentIndex(1)

            self.populate_detection_table(result.get("detections", []))
            timing = result.get("timing", {})
            t_infer = timing.get("infer_s", None)
            t_draw = timing.get("draw_s", None)
            timing_str = ""
            if t_infer is not None and t_draw is not None:
                timing_str = f" (infer {t_infer:.3f}s, draw {t_draw:.3f}s)"

            self.status_label.setText(f"完成：{Path(self.image_path).name} -> {out_img}{timing_str}")

            if self.auto_open_chk.isChecked():
                self.open_output_dir()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")

        # advance queue
        if self._queue:
            self._queue.pop(0)
        self._queue_done += 1
        self.update_progress(done=self._queue_done)
        self.save_settings()
        self.run_next_in_queue()

    def on_detection_error(self, error_msg):
        """检测错误"""
        QMessageBox.critical(self, "错误", error_msg)
        self.status_label.setText("检测失败")
        self._busy = False
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("🚀 开始检测（当前/队列）")

    def on_status_update(self, status):
        """状态更新"""
        self.status_label.setText(status)

    def populate_detection_table(self, detections):
        self.det_table.setRowCount(0)
        for det in detections or []:
            row = self.det_table.rowCount()
            self.det_table.insertRow(row)
            self.det_table.setItem(row, 0, QTableWidgetItem(str(det.get("class_id", ""))))
            self.det_table.setItem(row, 1, QTableWidgetItem(str(det.get("class_name", ""))))
            self.det_table.setItem(row, 2, QTableWidgetItem(f"{det.get('score', 0.0):.4f}"))
            self.det_table.setItem(row, 3, QTableWidgetItem(str(det.get("bbox_xyxy", ""))))
        self.tab_widget.setCurrentIndex(2)

    def check_server_health(self):
        url = self.server_input.text().rstrip("/") + "/health"
        try:
            req = Request(url, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok", False):
                self.status_label.setText("服务器连接正常 ✅")
                QMessageBox.information(self, "连接正常", f"已连接：{url}")
            else:
                raise Exception(data)
        except Exception as e:
            QMessageBox.warning(self, "连接失败", f"无法连接到：{url}\n\n{str(e)}")
            self.status_label.setText("服务器连接失败 ❌")

    def copy_ssh_command(self):
        ssh_cmd = "ssh -L 18080:127.0.0.1:18080 <user>@<server_host>"
        QApplication.clipboard().setText(ssh_cmd)
        self.status_label.setText("已复制SSH端口转发命令到剪贴板")

    def open_output_dir(self):
        out_dir = Path(self.output_input.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        QUrl.fromLocalFile(str(out_dir))
        try:
            from PyQt5.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
        except Exception:
            pass

    def open_last_result_image(self):
        key = str(Path(self.image_path)) if self.image_path else ""
        if not key or key not in self._last_saved_paths:
            QMessageBox.information(self, "提示", "当前没有可打开的结果图片。")
            return
        out_img, _ = self._last_saved_paths[key]
        try:
            from PyQt5.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(out_img))
        except Exception:
            pass

    def on_auto_open_changed(self, _state):
        self.auto_open_output = self.auto_open_chk.isChecked()
        self.save_settings()

    def load_settings(self):
        self.server_url = self._settings.value("server_url", self.server_url, type=str)
        self.output_dir = self._settings.value("output_dir", self.output_dir, type=str)
        self.score_threshold = float(self._settings.value("score_threshold", self.score_threshold))
        self.auto_open_output = bool(int(self._settings.value("auto_open_output", int(self.auto_open_output))))

        self.server_input.setText(self.server_url)
        self.output_input.setText(self.output_dir)
        self.score_slider.setValue(int(self.score_threshold * 100))
        self.auto_open_chk.setChecked(self.auto_open_output)

    def save_settings(self):
        self._settings.setValue("server_url", self.server_input.text())
        self._settings.setValue("output_dir", self.output_input.text())
        self._settings.setValue("score_threshold", float(self.score_threshold))
        self._settings.setValue("auto_open_output", int(self.auto_open_chk.isChecked()))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = []
        for u in urls:
            p = u.toLocalFile()
            if p:
                paths.append(p)
        if paths:
            if not self.image_path:
                self.image_path = paths[0]
                self.file_label.setText(Path(self.image_path).name)
                self.file_label.setStyleSheet("color: #333; font-weight: bold;")
                self.display_image(self.image_path, self.original_label)
            self.enqueue_images(paths, set_current=False)
            self.status_label.setText(f"拖拽加入队列: {len(paths)} 张")
            self.save_settings()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageDetectionApp()
    window.show()
    sys.exit(app.exec_())

import os
import sys
from pathlib import Path

import cv2
import dlib
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)


class DetectionWorker(QThread):
    """在独立线程中执行人脸检测，避免阻塞 UI 主线程"""
    result_ready = pyqtSignal(object, str)  # 发出 (overlay图像, 信息文本)

    def __init__(self):
        super().__init__()
        # 每次 submit 时从主线程拷入任务参数
        self._frame = None
        self._hog = None
        self._cnn = None
        self._predictor = None
        self._rec_model = None
        self._use_cnn = False
        self._enable_rec = False

    def submit(self, frame, hog, cnn, predictor, rec_model, use_cnn, enable_rec):
        """提交检测任务。若上一帧仍在处理则直接丢弃，保证实时性"""
        if self.isRunning():
            return  # 丢帧：避免任务堆积导致卡死
        self._frame     = frame.copy()
        self._hog       = hog
        self._cnn       = cnn
        self._predictor = predictor
        self._rec_model = rec_model
        self._use_cnn   = use_cnn
        self._enable_rec = enable_rec
        self.start()  # 启动线程（自动调用 run）

    def run(self):
        """线程体：执行检测并绘制，完成后发出信号"""
        img    = self._frame
        output = img.copy()
        gray   = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        rgb    = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # 识别模型需要 RGB
        info_lines = []

        # ── 步骤1：人脸检测（HOG 或 CNN）──────────────────────────
        if self._use_cnn and self._cnn is not None:
            # CNN 检测器输入 RGB，返回 mmod_rectangles，需取 .rect
            rects = [r.rect for r in self._cnn(rgb, 1)]
            info_lines.append(f"[CNN检测] 检测到 {len(rects)} 张人脸")
        else:
            # HOG 检测器输入灰度图，直接返回矩形列表
            rects = list(self._hog(gray, 1))
            info_lines.append(f"[HOG检测] 检测到 {len(rects)} 张人脸")

        # 动态缩放系数：以图像长边为基准，保证框/点在不同分辨率下视觉大小一致
        long_side = max(output.shape[:2])
        scale     = long_side / 1000.0                        # 基准：1000px 长边
        thickness = max(1, int(round(3 * scale)))             # 框线粗细
        dot_r     = max(1, int(round(4 * scale)))             # 关键点半径
        font_sc   = max(0.4, round(0.8 * scale, 1))          # 字体大小
        font_th   = max(1, int(round(2 * scale)))             # 字体粗细

        # ── 步骤2：逐人脸绘制框、关键点、识别特征 ─────────────────────────────
        for idx, rect in enumerate(rects):
            x1 = max(0, rect.left())
            y1 = max(0, rect.top())
            x2 = min(output.shape[1], rect.right())
            y2 = min(output.shape[0], rect.bottom())

            # 绘制人脸框（粗细随分辨率自适应）和编号
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), thickness)
            cv2.putText(output, f"Face {idx+1}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sc, (0, 255, 255), font_th)

            if self._predictor is not None:
                shape = self._predictor(gray, rect)  # 预测关键点
                num_parts = shape.num_parts           # 自动兼容5点/68点
                for i in range(num_parts):
                    x, y = shape.part(i).x, shape.part(i).y
                    cv2.circle(output, (x, y), dot_r, (0, 0, 255), -1)  # 红色关键点（半径自适应）
                info_lines.append(f"  Face {idx+1}: {num_parts} 个关键点")

                # 人脸识别：提取128维特征向量
                if self._rec_model is not None and self._enable_rec:
                    try:
                        face_chip = dlib.get_face_chip(rgb, shape)  # 对齐芯片
                        descriptor = np.array(
                            self._rec_model.compute_face_descriptor(face_chip)
                        )
                        norm = float(np.linalg.norm(descriptor))
                        info_lines.append(f"  Face {idx+1}: 特征已提取，范数={norm:.3f}")
                        cv2.putText(output, "Rec OK", (x1, y2 + int(35 * scale)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_sc, (255, 180, 0), font_th)
                    except Exception as e:
                        info_lines.append(f"  Face {idx+1}: 识别失败 - {e}")
            else:
                info_lines.append(f"  Face {idx+1}: 未加载关键点模型")

        # 检测完成，通知主线程更新 UI（跨线程用信号，安全）
        self.result_ready.emit(output, "\n".join(info_lines))


class ZoomableView(QGraphicsView):
    """支持滚轮缩放的 QGraphicsView，Ctrl+滚轮或直接滚轮均可缩放"""
    def wheelEvent(self, event):
        # 滚轮向上放大，向下缩小；每格缩放 15%
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


def read_image(path):
    # 使用 fromfile 兼容中文路径
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, img):
    # 使用 imencode + tofile 兼容中文路径
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    success, buffer = cv2.imencode(ext, img)
    if not success:
        return False
    buffer.tofile(path)
    return True


class FaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别可视化")
        self.resize(1200, 800)

        # ── 三类模型实例 ──────────────────────────────────────────
        self.hog_detector = dlib.get_frontal_face_detector()   # HOG人脸检测器（快速）
        self.cnn_detector = None                                # CNN人脸检测器（mmod，更准确）
        self.predictor = None                                   # 关键点检测模型（5点/68点）
        self.face_rec_model = None                              # 人脸识别模型（128维特征）
        self.use_cnn = False                                    # 当前是否使用CNN检测器

        self.current_image_raw = None
        self.current_image_overlay = None
        self.last_frame_raw = None
        self.last_frame_overlay = None

        self.show_results = True
        self.video_cap = None
        self._auto_fit = True  # 打开新文件时自动适应窗口，缩放时不重置

        # ── 检测工作线程 ──────────────────────────────────────────
        self.worker = DetectionWorker()
        self.worker.result_ready.connect(self.on_result_ready)  # 结果回调到主线程

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.scene = QGraphicsScene(self)
        self.view = ZoomableView(self.scene)  # 支持滚轮缩放的视图
        self.view.setRenderHint(QPainter.Antialiasing)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.stretch_check = QCheckBox("拉伸填充")
        self.stretch_check.stateChanged.connect(self.fit_in_view)

        self.init_ui()
        self.load_models()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        controls = QWidget()
        controls.setFixedWidth(290)
        controls_layout = QVBoxLayout(controls)

        # ── 检测器选择：HOG（快速） vs CNN-mmod（更准确）────────────
        detector_group = QGroupBox("人脸检测器")
        detector_layout = QVBoxLayout(detector_group)
        self.hog_radio = QRadioButton("HOG（快速，默认）")
        self.cnn_radio = QRadioButton("CNN-mmod（更准确，较慢）")
        self.hog_radio.setChecked(True)
        self.hog_radio.toggled.connect(self.on_detector_changed)  # 切换时重新检测
        self.cnn_status_label = QLabel("mmod模型：未加载")
        self.cnn_status_label.setStyleSheet("color: gray; font-size: 11px;")
        detector_layout.addWidget(self.hog_radio)
        detector_layout.addWidget(self.cnn_radio)
        detector_layout.addWidget(self.cnn_status_label)

        # ── 关键点模型选择（shape_predictor_*.dat）────────────────
        landmark_group = QGroupBox("关键点模型")
        landmark_layout = QHBoxLayout(landmark_group)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(48)
        refresh_btn.clicked.connect(self.load_models)
        landmark_layout.addWidget(self.model_combo)
        landmark_layout.addWidget(refresh_btn)

        # ── 人脸识别模型开关（dlib_face_recognition_resnet_model_v1.dat）
        rec_group = QGroupBox("人脸识别模型（128维特征）")
        rec_layout = QVBoxLayout(rec_group)
        self.rec_check = QCheckBox("启用特征提取")
        self.rec_check.setEnabled(False)  # 默认禁用，加载成功后才启用
        self.rec_check.stateChanged.connect(self.update_image_display)
        self.rec_status_label = QLabel("识别模型：未加载")
        self.rec_status_label.setStyleSheet("color: gray; font-size: 11px;")
        rec_layout.addWidget(self.rec_check)
        rec_layout.addWidget(self.rec_status_label)

        # ── 输入 ──────────────────────────────────────────────────
        source_group = QGroupBox("输入")
        source_layout = QGridLayout(source_group)
        open_img_btn = QPushButton("打开图片")
        open_img_btn.clicked.connect(self.open_image)
        open_video_btn = QPushButton("打开视频")
        open_video_btn.clicked.connect(self.open_video)
        stop_video_btn = QPushButton("停止视频")
        stop_video_btn.clicked.connect(self.stop_video)
        source_layout.addWidget(open_img_btn, 0, 0)
        source_layout.addWidget(open_video_btn, 0, 1)
        source_layout.addWidget(stop_video_btn, 1, 0, 1, 2)

        # ── 显示控制 ──────────────────────────────────────────────
        view_group = QGroupBox("显示")
        view_layout = QGridLayout(view_group)
        toggle_btn = QPushButton("切换结果框/点")
        toggle_btn.clicked.connect(self.toggle_overlay)
        zoom_in_btn = QPushButton("放大")
        zoom_in_btn.clicked.connect(lambda: self.zoom(1.2))
        zoom_out_btn = QPushButton("缩小")
        zoom_out_btn.clicked.connect(lambda: self.zoom(1 / 1.2))
        fit_btn = QPushButton("适应窗口")
        fit_btn.clicked.connect(self.fit_in_view)
        view_layout.addWidget(toggle_btn, 0, 0, 1, 2)
        view_layout.addWidget(zoom_in_btn, 1, 0)
        view_layout.addWidget(zoom_out_btn, 1, 1)
        view_layout.addWidget(fit_btn, 2, 0)
        view_layout.addWidget(self.stretch_check, 2, 1)

        # ── 保存 ──────────────────────────────────────────────────
        save_group = QGroupBox("保存")
        save_layout = QGridLayout(save_group)
        save_with_btn = QPushButton("保存带框")
        save_with_btn.clicked.connect(self.save_with_overlay)
        save_without_btn = QPushButton("保存无框")
        save_without_btn.clicked.connect(self.save_without_overlay)
        save_layout.addWidget(save_with_btn, 0, 0)
        save_layout.addWidget(save_without_btn, 0, 1)

        self.status_label = QLabel("就绪")
        # 检测结果信息面板，显示人脸数量、关键点数、特征提取状态
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(110)
        self.info_text.setPlaceholderText("检测结果信息...")

        controls_layout.addWidget(detector_group)
        controls_layout.addWidget(landmark_group)
        controls_layout.addWidget(rec_group)
        controls_layout.addWidget(source_group)
        controls_layout.addWidget(view_group)
        controls_layout.addWidget(save_group)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.info_text)
        controls_layout.addStretch(1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)

    def load_models(self):
        model_dir = Path("pt")

        # 1. 加载 CNN 人脸检测器（mmod_human_face_detector.dat）
        #    用于替代 HOG 检测器，对非正面、遮挡人脸效果更好，但速度较慢
        cnn_path = model_dir / "mmod_human_face_detector.dat"
        if cnn_path.exists():
            try:
                self.cnn_detector = dlib.cnn_face_detection_model_v1(str(cnn_path))
                self.cnn_status_label.setText("mmod模型：已加载 ✓")
                self.cnn_status_label.setStyleSheet("color: green; font-size: 11px;")
                self.cnn_radio.setEnabled(True)
            except Exception as e:
                self.cnn_detector = None
                self.cnn_status_label.setText(f"mmod加载失败: {e}")
                self.cnn_radio.setEnabled(False)
        else:
            self.cnn_radio.setEnabled(False)
            self.cnn_status_label.setText("mmod模型：文件不存在")

        # 2. 加载人脸识别模型（dlib_face_recognition_resnet_model_v1.dat）
        #    输入对齐人脸，输出128维特征向量，可用于人脸比对
        rec_path = model_dir / "dlib_face_recognition_resnet_model_v1.dat"
        if rec_path.exists():
            try:
                self.face_rec_model = dlib.face_recognition_model_v1(str(rec_path))
                self.rec_status_label.setText("识别模型：已加载 ✓")
                self.rec_status_label.setStyleSheet("color: green; font-size: 11px;")
                self.rec_check.setEnabled(True)
            except Exception as e:
                self.face_rec_model = None
                self.rec_status_label.setText(f"识别模型加载失败: {e}")
        else:
            self.rec_status_label.setText("识别模型：文件不存在")

        # 3. 加载关键点检测模型（shape_predictor_*.dat），填充下拉框
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        landmark_files = sorted(model_dir.glob("shape_predictor_*.dat"))
        if not landmark_files:
            self.model_combo.addItem("未找到关键点模型", None)
            self.predictor = None
        else:
            for file_path in landmark_files:
                label = file_path.name
                if "5" in label:
                    label += "  [5点]"
                elif "68" in label:
                    label += "  [68点]"
                self.model_combo.addItem(label, str(file_path))
            self.load_predictor(self.model_combo.currentData())
        self.model_combo.blockSignals(False)

    def load_predictor(self, model_path):
        """加载关键点检测器 shape_predictor"""
        if not model_path:
            return
        try:
            self.predictor = dlib.shape_predictor(model_path)
            self.status_label.setText(f"关键点模型: {Path(model_path).name}")
        except Exception as exc:
            self.predictor = None
            QMessageBox.warning(self, "关键点模型加载失败", str(exc))

    def on_model_changed(self, index):
        model_path = self.model_combo.itemData(index)
        if model_path:
            self.load_predictor(model_path)

    def on_detector_changed(self):
        """切换 HOG / CNN 检测器"""
        self.use_cnn = self.cnn_radio.isChecked()
        if self.use_cnn and self.cnn_detector is None:
            QMessageBox.warning(self, "提示", "CNN检测器未加载，请确认 mmod_human_face_detector.dat 存在于 pt/ 目录")
            self.hog_radio.setChecked(True)
            return
        self.update_image_display()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.png *.jpeg *.bmp)"
        )
        if not file_path:
            return
        img = read_image(file_path)
        if img is None:
            QMessageBox.warning(self, "读取失败", "无法读取图片")
            return
        self.stop_video()
        self.current_image_raw = img
        self._auto_fit = True  # 新图片打开时自动适应
        self.update_image_display()

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not file_path:
            return
        self.stop_video()
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "读取失败", "无法打开视频")
            return
        self.video_cap = cap
        self._auto_fit = True   # 新视频首帧自动适应
        self.timer.start(30)    # 每30ms读一帧，检测慢时自动丢帧
        self.status_label.setText("视频播放中")

    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()
        self.worker.wait()  # 等待当前帧检测完，避免线程使用已释放资源
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.status_label.setText("已停止视频")

    def _make_submit_args(self):
        """打包当前模型配置，供 worker.submit 使用"""
        return (
            self.hog_detector, self.cnn_detector,
            self.predictor, self.face_rec_model,
            self.use_cnn,
            self.rec_check.isChecked(),
        )

    def next_frame(self):
        """每 30ms 读取一帧；检测线程忙时直接丢弃，不阻塞 UI"""
        if self.video_cap is None:
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.stop_video()
            return
        self.last_frame_raw = frame  # 保存原始帧供保存使用
        if self.show_results:
            # 提交给工作线程检测（忙则丢帧）
            self.worker.submit(frame, *self._make_submit_args())
        else:
            # 无需检测，直接显示原图（不调用 fit_in_view，避免频繁重算）
            self._show_pixmap(frame)

    def update_image_display(self):
        """图片模式刷新，也走异步线程检测"""
        if self.current_image_raw is None:
            return
        if self.show_results:
            self._auto_fit = True
            self.worker.submit(self.current_image_raw, *self._make_submit_args())
        else:
            self.current_image_overlay = None
            self._auto_fit = True
            self.update_view(self.current_image_raw)

    def on_result_ready(self, overlay, info_text):
        """工作线程检测完成后，在主线程更新显示（PyQt 信号保证线程安全）"""
        self.info_text.setText(info_text)
        if self.timer.isActive():  # 视频模式
            self.last_frame_overlay = overlay
        else:                      # 图片模式
            self.current_image_overlay = overlay
        self.update_view(overlay)

    def _show_pixmap(self, img):
        """仅更新像素图，不重置缩放（视频无需每帧 fit）"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)

    def update_view(self, img):
        """更新显示，打开新文件时自动适应窗口；视频播放时不每帧 fit"""
        self._show_pixmap(img)
        if self._auto_fit:
            self.fit_in_view()
            self._auto_fit = False

    def fit_in_view(self):
        """适应窗口：重置缩放后 fitInView，保证任何时候点击或窗口拖拽都生效"""
        if self.pixmap_item.pixmap().isNull():
            return
        aspect = Qt.IgnoreAspectRatio if self.stretch_check.isChecked() else Qt.KeepAspectRatio
        self.view.resetTransform()
        self.view.fitInView(self.pixmap_item, aspect)

    def resizeEvent(self, event):
        """窗口大小改变时（包括拖拽缩放）自动让图像适应新视图尺寸"""
        super().resizeEvent(event)
        self.fit_in_view()

    def zoom(self, factor):
        if self.pixmap_item.pixmap().isNull():
            return
        self.view.scale(factor, factor)

    def toggle_overlay(self):
        """切换结果框/点显示，图片和视频模式下均即时刷新当前帧"""
        self.show_results = not self.show_results
        if self.timer.isActive():
            # 视频模式：用缓存的最后一帧即时切换，不等待下一帧
            if not self.show_results and self.last_frame_raw is not None:
                self._show_pixmap(self.last_frame_raw)  # 显示无框原图
            elif self.show_results and self.last_frame_overlay is not None:
                self._show_pixmap(self.last_frame_overlay)  # 显示上一次带框结果
            return
        self.update_image_display()

    def save_with_overlay(self):
        if self.timer.isActive():
            img = self.last_frame_overlay
        else:
            img = self.current_image_overlay
        if img is None:
            QMessageBox.information(self, "提示", "当前没有带框结果可保存")
            return
        self.save_image(img)

    def save_without_overlay(self):
        if self.timer.isActive():
            img = self.last_frame_raw
        else:
            img = self.current_image_raw
        if img is None:
            QMessageBox.information(self, "提示", "当前没有原图可保存")
            return
        self.save_image(img)

    def save_image(self, img):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", "", "Image Files (*.jpg *.png *.jpeg *.bmp)"
        )
        if not file_path:
            return
        if not write_image(file_path, img):
            QMessageBox.warning(self, "保存失败", "图片保存失败")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec_())

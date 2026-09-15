"""模式5标记点检测参数调试工具。

该工具直接使用 TouchSensor 的预处理帧，并复用 SDK 中的统一 BGR 阈值、
形态学闭运算和 HoughCircles 检测流程。界面会同步显示原图、二值 mask 和
圆检测结果，参数修改后立即作用于当前帧。

运行方式：
    python samples/demo_mode5_marker_detection_Qt.py 0

快捷键：
    Space / P  暂停或继续取帧
    B          使用当前参数重新建立 SDK 标记点基线
    Q / Esc    退出
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.algorithm.touch_sensor import TouchSensor


class MarkerDetectionViewer(QtWidgets.QWidget):
    def __init__(self, sensor, interval_ms=33, parent=None):
        super().__init__(parent)
        self.sensor = sensor
        
        startup_config = self._get_marker_detection_config()
        self._startup_threshold = dict(startup_config["threshold"])
        self._startup_hough = dict(startup_config["hough"])
        self._startup_expected_count = startup_config["expected_count"]
        self.interval_ms = max(1, int(interval_ms))
        self._closed = False
        self._busy = False
        self._paused = False
        self._last_frame = None
        self._last_rgb_images = {}
        self._fps = 0.0
        self._fps_frames = 0
        self._fps_started = time.perf_counter()

        self.setWindowTitle("Sightac Mode 5 - Marker Detection Tuner")
        self.resize(1680, 820)

        self.raw_image_label = self._create_image_label()
        self.mask_image_label = self._create_image_label()
        self.result_image_label = self._create_image_label()

        image_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        image_splitter.setChildrenCollapsible(False)
        image_splitter.addWidget(
            self._create_image_group("原始图像（检测输入）", self.raw_image_label)
        )
        image_splitter.addWidget(
            self._create_image_group("统一阈值二值 Mask", self.mask_image_label)
        )
        image_splitter.addWidget(
            self._create_image_group("Hough 圆检测结果", self.result_image_label)
        )
        for index in range(3):
            image_splitter.setStretchFactor(index, 1)

        self.b_min_spin = self._create_int_spin(0, 255)
        self.g_min_spin = self._create_int_spin(0, 255)
        self.r_min_spin = self._create_int_spin(0, 255)

        self.dp_spin = self._create_double_spin(1.0, 10.0, 0.1, 1)
        self.min_dist_spin = self._create_double_spin(1.0, 300.0, 1.0, 1)
        self.param1_spin = self._create_double_spin(1.0, 500.0, 1.0, 1)
        self.param2_spin = self._create_double_spin(0.1, 200.0, 0.5, 1)
        self.min_radius_spin = self._create_int_spin(0, 200)
        self.max_radius_spin = self._create_int_spin(1, 300)

        self.expected_count_spin = self._create_int_spin(0, 1000)
        self.expected_count_spin.setSpecialValueText("不限制")

        self._load_parameters_from_sdk()

        threshold_group = QtWidgets.QGroupBox("全图统一 BGR 阈值")
        threshold_form = QtWidgets.QFormLayout(threshold_group)
        threshold_form.addRow("B 最小值", self.b_min_spin)
        threshold_form.addRow("G 最小值", self.g_min_spin)
        threshold_form.addRow("R 最小值", self.r_min_spin)

        hough_group = QtWidgets.QGroupBox("HoughCircles 参数")
        hough_form = QtWidgets.QFormLayout(hough_group)
        hough_form.addRow("dp", self.dp_spin)
        hough_form.addRow("minDist", self.min_dist_spin)
        hough_form.addRow("param1", self.param1_spin)
        hough_form.addRow("param2", self.param2_spin)
        hough_form.addRow("minRadius", self.min_radius_spin)
        hough_form.addRow("maxRadius", self.max_radius_spin)
        hough_form.addRow("期望圆数量", self.expected_count_spin)

        morphology_note = QtWidgets.QLabel(
            "Mask 后处理与 SDK 保持一致：5×5 矩形核，闭运算 1 次。\n"
            "参数改变会立即更新当前检测；按 B 才会重建 SDK 追踪基线。"
        )
        morphology_note.setWordWrap(True)

        self.pause_button = QtWidgets.QPushButton("暂停画面 (Space)")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._set_paused)

        self.reload_button = QtWidgets.QPushButton("恢复启动参数")
        self.reload_button.clicked.connect(self._reload_parameters)

        self.calibrate_button = QtWidgets.QPushButton("当前参数重建基线 (B)")
        self.calibrate_button.clicked.connect(self.calibrate_baseline)

        self.quit_button = QtWidgets.QPushButton("退出")
        self.quit_button.clicked.connect(self.close)

        self.parameter_preview = QtWidgets.QPlainTextEdit()
        self.parameter_preview.setReadOnly(True)
        self.parameter_preview.setMaximumHeight(185)
        self.parameter_preview.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        controls_widget = QtWidgets.QWidget()
        controls_widget.setMinimumWidth(285)
        controls_widget.setMaximumWidth(360)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(threshold_group)
        controls_layout.addWidget(hough_group)
        controls_layout.addWidget(morphology_note)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.reload_button)
        controls_layout.addWidget(self.calibrate_button)
        controls_layout.addWidget(self.quit_button)
        controls_layout.addWidget(QtWidgets.QLabel("当前参数代码"))
        controls_layout.addWidget(self.parameter_preview)
        controls_layout.addStretch(1)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setWidget(controls_widget)

        self.status_label = QtWidgets.QLabel("等待传感器图像...")
        self.status_label.setWordWrap(True)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.addWidget(controls_scroll)
        content_layout.addWidget(image_splitter, stretch=1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(content_layout, stretch=1)
        layout.addWidget(self.status_label)

        self._parameter_widgets = (
            self.b_min_spin,
            self.g_min_spin,
            self.r_min_spin,
            self.dp_spin,
            self.min_dist_spin,
            self.param1_spin,
            self.param2_spin,
            self.min_radius_spin,
            self.max_radius_spin,
            self.expected_count_spin,
        )
        for widget in self._parameter_widgets:
            widget.valueChanged.connect(self._parameters_changed)

        self._parameters_changed()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.interval_ms)

    @staticmethod
    def _create_image_label():
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(280, 240)
        label.setStyleSheet("background: #111; border: 1px solid #444;")
        return label

    @staticmethod
    def _create_image_group(title, image_label):
        group = QtWidgets.QGroupBox(title)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.addWidget(image_label)
        return group

    @staticmethod
    def _create_int_spin(minimum, maximum):
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        return spin

    @staticmethod
    def _create_double_spin(minimum, maximum, step, decimals):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _load_parameters_from_sdk(self):
        config = self._get_marker_detection_config()
        self._set_parameter_widgets(
            config["threshold"],
            config["hough"],
            config["expected_count"],
        )

    def _get_marker_detection_config(self):
        return {
            "threshold": dict(self.sensor.marker_bgr_threshold),
            "hough": dict(self.sensor.marker_hough_params),
            "expected_count": self.sensor.marker_expected_count,
        }

    def _set_marker_detection_config(
        self,
        threshold,
        hough,
        expected_count=None,
    ):
        self.sensor.marker_bgr_threshold = dict(threshold)
        self.sensor.marker_hough_params = dict(hough)
        self.sensor.marker_expected_count = expected_count

    def _get_marker_detection_debug_data(self, frame=None):
        if frame is None:
            self.sensor.update_frame()
            frame = self.sensor.get_raw_frame()
        if frame is None:
            raise RuntimeError("无法获取标记点检测输入图像")
        return {
            "frame": frame,
            "mask": self.sensor._build_marker_mask(frame),
        }

    def _set_parameter_widgets(self, threshold, hough, expected_count):

        self.b_min_spin.setValue(int(threshold["b_min"]))
        self.g_min_spin.setValue(int(threshold["g_min"]))
        self.r_min_spin.setValue(int(threshold["r_min"]))
        self.dp_spin.setValue(float(hough["dp"]))
        self.min_dist_spin.setValue(float(hough["minDist"]))
        self.param1_spin.setValue(float(hough["param1"]))
        self.param2_spin.setValue(float(hough["param2"]))
        self.min_radius_spin.setValue(int(hough["minRadius"]))
        self.max_radius_spin.setValue(int(hough["maxRadius"]))
        self.expected_count_spin.setValue(
            0 if expected_count is None else int(expected_count)
        )

    def _reload_parameters(self):
        blockers = [QtCore.QSignalBlocker(item) for item in self._parameter_widgets]
        self._set_parameter_widgets(
            self._startup_threshold,
            self._startup_hough,
            self._startup_expected_count,
        )
        del blockers
        self._parameters_changed()

    def _current_parameters(self):
        return {
            "threshold": {
                "b_min": self.b_min_spin.value(),
                "g_min": self.g_min_spin.value(),
                "r_min": self.r_min_spin.value(),
            },
            "hough": {
                "dp": self.dp_spin.value(),
                "minDist": self.min_dist_spin.value(),
                "param1": self.param1_spin.value(),
                "param2": self.param2_spin.value(),
                "minRadius": self.min_radius_spin.value(),
                "maxRadius": self.max_radius_spin.value(),
            },
            "expected_count": self.expected_count_spin.value(),
        }

    def _parameters_changed(self):
        params = self._current_parameters()
        self._set_marker_detection_config(
            threshold=params["threshold"],
            hough=params["hough"],
            expected_count=(
                None
                if params["expected_count"] == 0
                else params["expected_count"]
            ),
        )
        self._update_parameter_preview(params)
        if self._last_frame is not None and not self._busy:
            self._process_frame(self._last_frame)

    def _update_parameter_preview(self, params):
        threshold = params["threshold"]
        hough = params["hough"]
        expected_count = params["expected_count"]
        expected_text = "None" if expected_count == 0 else str(expected_count)
        self.parameter_preview.setPlainText(
            "self.marker_bgr_threshold = {\n"
            f"    \"b_min\": {threshold['b_min']},\n"
            f"    \"g_min\": {threshold['g_min']},\n"
            f"    \"r_min\": {threshold['r_min']},\n"
            "}\n"
            "self.marker_hough_params = {\n"
            f"    \"dp\": {hough['dp']:.1f}, \"minDist\": {hough['minDist']:.1f},\n"
            f"    \"param1\": {hough['param1']:.1f}, \"param2\": {hough['param2']:.1f},\n"
            f"    \"minRadius\": {hough['minRadius']}, \"maxRadius\": {hough['maxRadius']},\n"
            "}\n"
            f"self.marker_expected_count = {expected_text}"
        )

    def update_frame(self):
        if self._busy or self._closed or self._paused:
            return

        self._busy = True
        try:
            debug_data = self._get_marker_detection_debug_data()
            frame = debug_data["frame"]
            self._last_frame = frame.copy()
            self._process_frame(
                self._last_frame,
                marker_mask=debug_data["mask"],
            )
        except Exception as exc:
            self.timer.stop()
            traceback.print_exc()
            self.status_label.setText(f"错误：{exc}")
            QtWidgets.QMessageBox.critical(self, "标记点检测失败", str(exc))
        finally:
            self._busy = False

    def _process_frame(self, frame, marker_mask=None):
        params = self._current_parameters()
        min_radius = params["hough"]["minRadius"]
        max_radius = params["hough"]["maxRadius"]
        if min_radius > max_radius:
            self.status_label.setText("参数错误：minRadius 不能大于 maxRadius")
            return

        if marker_mask is None:
            marker_mask = self._get_marker_detection_debug_data(
                frame=frame,
            )["mask"]
        circles = cv2.HoughCircles(
            marker_mask,
            cv2.HOUGH_GRADIENT,
            **params["hough"],
        )

        detected_circles = []
        if circles is not None:
            detected_circles = np.rint(circles[0]).astype(np.int32).tolist()
            detected_circles.sort(key=lambda item: (item[1], item[0]))

        result = frame.copy()
        for index, (x, y, radius) in enumerate(detected_circles):
            cv2.circle(result, (x, y), radius, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(result, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                result,
                str(index + 1),
                (x + radius + 3, y - radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        mask_bgr = cv2.cvtColor(marker_mask, cv2.COLOR_GRAY2BGR)
        self._set_bgr_image("raw", self.raw_image_label, frame)
        self._set_bgr_image("mask", self.mask_image_label, mask_bgr)
        self._set_bgr_image("result", self.result_image_label, result)
        self._update_status(frame, detected_circles, params)

    def _update_status(self, frame, circles, params):
        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_started
        if elapsed >= 1.0:
            self._fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_started = now

        expected_count = params["expected_count"]
        if expected_count == 0:
            count_text = f"检测到 {len(circles)} 个圆"
            count_style = ""
        elif len(circles) == expected_count:
            count_text = f"检测到 {len(circles)}/{expected_count} 个圆，数量正确"
            count_style = "color: #16803a; font-weight: bold;"
        else:
            count_text = f"检测到 {len(circles)}/{expected_count} 个圆，数量不匹配"
            count_style = "color: #c62828; font-weight: bold;"

        state = "已暂停" if self._paused else "实时"
        height, width = frame.shape[:2]
        self.status_label.setStyleSheet(count_style)
        self.status_label.setText(
            f"{state}  |  图像 {width}×{height}  |  FPS {self._fps:.1f}  |  "
            f"{count_text}"
        )

    def _set_bgr_image(self, name, label, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._last_rgb_images[name] = np.ascontiguousarray(rgb)
        self._refresh_pixmap(label, self._last_rgb_images[name])

    @staticmethod
    def _refresh_pixmap(label, image):
        height, width = image.shape[:2]
        qimage = QtGui.QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        label.setPixmap(
            QtGui.QPixmap.fromImage(qimage).scaled(
                label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _refresh_pixmaps(self):
        labels = {
            "raw": self.raw_image_label,
            "mask": self.mask_image_label,
            "result": self.result_image_label,
        }
        for name, image in self._last_rgb_images.items():
            self._refresh_pixmap(labels[name], image)

    def _set_paused(self, paused):
        self._paused = bool(paused)
        self.pause_button.setText(
            "继续画面 (Space)" if self._paused else "暂停画面 (Space)"
        )
        if self._last_frame is not None:
            self._process_frame(self._last_frame)

    def calibrate_baseline(self):
        if self._busy or self._closed:
            return

        self._busy = True
        self.status_label.setStyleSheet("")
        self.status_label.setText("正在使用当前参数重建标记点基线，请保持传感器空载...")
        QtWidgets.QApplication.processEvents()
        try:
            result = self.sensor.calibrate_baseline(
                calibration_frames=5,
                mode=5,
            )
            marker_count = len(result["marker_data"]["first_centers"])
            self.status_label.setText(f"基线重建完成，共建立 {marker_count} 个模板")
        except Exception as exc:
            traceback.print_exc()
            self.status_label.setText(f"基线重建失败：{exc}")
            QtWidgets.QMessageBox.critical(self, "基线重建失败", str(exc))
        finally:
            self._busy = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmaps()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return
        if event.key() in (QtCore.Qt.Key_Space, QtCore.Qt.Key_P):
            self.pause_button.toggle()
            return
        if event.key() == QtCore.Qt.Key_B:
            self.calibrate_baseline()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self.timer.stop()
            self.sensor.release()
        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(
        description="模式5标记点二值化和霍夫圆检测 Qt 调参工具"
    )
    parser.add_argument(
        "usb_id",
        nargs="?",
        type=int,
        default=0,
        help="摄像头 USB ID，默认 0",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=33,
        help="界面更新间隔，默认 33ms",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        [sys.argv[0]]
    )

    try:
        sensor = TouchSensor(usb_id=args.usb_id)
    except Exception as exc:
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(None, "传感器连接失败", str(exc))
        return 1

    viewer = MarkerDetectionViewer(sensor, interval_ms=args.interval_ms)
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

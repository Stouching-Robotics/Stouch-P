"""模式5可视化：Hue 热力图 + 标记点位移白色箭头。

运行方式：
    python samples/demo_mode5_marker_hue_Qt.py 0

快捷键：
    B          重新校准 Hue 基线和标记点模板
    Q / Esc    退出
"""

from __future__ import annotations

import argparse
import sys
import threading
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


DEFAULT_HUE_MAX = 30.0
DEFAULT_ARROW_SCALE = 4.0


def hue_to_heatmap(hue_matrix, hue_max):
    """按 Hue 变化强度映射热力图：无变化为蓝，强接触为红。"""
    if hue_matrix is None:
        return None

    hue = np.asarray(hue_matrix, dtype=np.float32)
    if hue.ndim != 2 or hue.size == 0:
        return None

    hue = np.nan_to_num(hue, nan=0.0, posinf=0.0, neginf=0.0)
    hue_max = max(float(hue_max), 1e-6)
    hue_magnitude = np.abs(hue)
    normalized = np.clip(
        hue_magnitude * (255.0 / hue_max),
        0.0,
        255.0,
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def draw_marker_overlay(heatmap, marker_data, arrow_scale):
    """在热力图上只绘制白色标记点位移箭头。"""
    if heatmap is None:
        return None

    canvas = heatmap.copy()
    first_centers = marker_data.get("first_centers", [])
    displacements = marker_data.get("displacements", [])

    for index, first_center in enumerate(first_centers):
        start = (int(first_center[0]), int(first_center[1]))
        displacement = (
            displacements[index]
            if index < len(displacements)
            else None
        )

        if displacement is None:
            continue

        dx, dy = float(displacement[0]), float(displacement[1])
        end = (
            int(round(start[0] + dx * arrow_scale)),
            int(round(start[1] + dy * arrow_scale)),
        )

        
        if np.hypot(dx, dy) >= 0.5:
            cv2.arrowedLine(
                canvas,
                start,
                end,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
                tipLength=0.30,
            )
            cv2.arrowedLine(
                canvas,
                start,
                end,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.30,
            )

    return canvas


class Mode5Worker(QtCore.QThread):
    """后台独占 TouchSensor，并只向 GUI 通知“有最新帧可取”。"""

    frame_available = QtCore.pyqtSignal()
    initialized = QtCore.pyqtSignal(str)
    calibration_finished = QtCore.pyqtSignal(bool, str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        usb_id,
        interval_ms=0,
        parent=None,
    ):
        super().__init__(parent)
        self.usb_id = int(usb_id)
        
        self.minimum_cycle_s = max(0, int(interval_ms)) / 1000.0

        self._latest_packet = None
        self._packet_lock = threading.Lock()
        self._notification_pending = False

        self._settings_lock = threading.Lock()
        self._hue_max = DEFAULT_HUE_MAX
        self._arrow_scale = DEFAULT_ARROW_SCALE

        self._calibration_requested = threading.Event()
        self._wake_event = threading.Event()
        self._pipeline_fps = 0.0
        self._fps_frames = 0
        self._fps_started = time.perf_counter()

    def set_visual_settings(self, hue_max, arrow_scale):
        """该方法可从 GUI 线程调用；工作线程每帧读取一次快照。"""
        with self._settings_lock:
            self._hue_max = float(hue_max)
            self._arrow_scale = float(arrow_scale)

    def request_calibration(self):
        self._calibration_requested.set()
        self._wake_event.set()

    def stop(self):
        self.requestInterruption()
        self._wake_event.set()

    def take_latest_packet(self):
        """取走最新结果并允许工作线程再次发送一次通知。"""
        with self._packet_lock:
            packet = self._latest_packet
            self._latest_packet = None
            self._notification_pending = False
        return packet

    def _publish_latest(self, packet):
        
        should_notify = False
        with self._packet_lock:
            self._latest_packet = packet
            if not self._notification_pending:
                self._notification_pending = True
                should_notify = True
        if should_notify:
            self.frame_available.emit()

    def _visual_settings(self):
        with self._settings_lock:
            return self._hue_max, self._arrow_scale

    def _reset_fps(self):
        self._pipeline_fps = 0.0
        self._fps_frames = 0
        self._fps_started = time.perf_counter()

    def _count_pipeline_frame(self):
        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_started
        if elapsed >= 1.0:
            self._pipeline_fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_started = now
        return self._pipeline_fps

    @staticmethod
    def _result_statistics(result):
        marker_data = result["marker_data"]
        displacements = marker_data.get("displacements", [])
        scores = marker_data.get("scores", [])
        valid = [item for item in displacements if item is not None]
        max_displacement = max(
            (float(np.hypot(item[0], item[1])) for item in valid),
            default=0.0,
        )
        valid_scores = [
            float(score)
            for score, displacement in zip(scores, displacements)
            if displacement is not None
        ]
        mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0

        hue = result["hue_matrix"]
        if hue is None or np.asarray(hue).size == 0:
            hue_abs_max = hue_abs_sum = 0.0
        else:
            hue_array = np.asarray(hue, dtype=np.float32)
            hue_magnitude = np.abs(
                np.nan_to_num(
                    hue_array,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            hue_abs_max = float(np.max(hue_magnitude))
            hue_abs_sum = float(np.sum(hue_magnitude, dtype=np.float64))

        return {
            "device_type": result["device_type"],
            "valid_markers": len(valid),
            "total_markers": len(marker_data.get("first_centers", [])),
            "max_displacement": max_displacement,
            "mean_score": mean_score,
            "hue_abs_max": hue_abs_max,
            "hue_abs_sum": hue_abs_sum,
            "fz_total": float(result["fz_total"] or 0.0),
        }

    def _build_packet(self, result, sensor_ms):
        hue_max, arrow_scale = self._visual_settings()
        visual_started = time.perf_counter()

        heatmap = hue_to_heatmap(result["hue_matrix"], hue_max)
        display = draw_marker_overlay(
            heatmap,
            result["marker_data"],
            arrow_scale,
        )
        if display is None:
            raise RuntimeError("模式5没有生成可显示的 Hue 图像")

        rgb_image = np.ascontiguousarray(
            cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        )

        stats = self._result_statistics(result)
        stats["sensor_ms"] = sensor_ms
        stats["visual_ms"] = (
            time.perf_counter() - visual_started
        ) * 1000.0
        stats["pipeline_fps"] = self._count_pipeline_frame()
        return {"rgb_image": rgb_image, "stats": stats}

    def _calibrate(self, sensor):
        self._calibration_requested.clear()
        try:
            result = sensor.calibrate_baseline(
                calibration_frames=5,
                mode=5,
            )
            marker_count = len(
                result["marker_data"].get("first_centers", [])
            )
            self._reset_fps()
            self.calibration_finished.emit(
                True,
                f"校准完成，建立了 {marker_count} 个标记点模板",
            )
        except BaseException as exc:
            traceback.print_exc()
            self.calibration_finished.emit(False, str(exc))

    def run(self):
        sensor = None
        try:
            
            sensor = TouchSensor(usb_id=self.usb_id)
            if self.isInterruptionRequested():
                return
            self.initialized.emit(sensor.device_type)

            while not self.isInterruptionRequested():
                if self._calibration_requested.is_set():
                    self._wake_event.clear()
                    self._calibrate(sensor)
                    if self.isInterruptionRequested():
                        break

                cycle_started = time.perf_counter()
                sensor_started = cycle_started
                result = sensor.update(5)
                sensor_elapsed = time.perf_counter() - sensor_started
                sensor_ms = sensor_elapsed * 1000.0
                packet = self._build_packet(result, sensor_ms)

                self._publish_latest(packet)

                remaining = self.minimum_cycle_s - (
                    time.perf_counter() - cycle_started
                )
                if remaining > 0:
                    self._wake_event.wait(remaining)
                    self._wake_event.clear()
        except BaseException as exc:
            traceback.print_exc()
            if not self.isInterruptionRequested():
                self.error_occurred.emit(str(exc))
        finally:
            if sensor is not None:
                try:
                    sensor.release()
                except Exception:
                    traceback.print_exc()


class Mode5Viewer(QtWidgets.QWidget):
    def __init__(
        self,
        usb_id,
        interval_ms=0,
        parent=None,
    ):
        super().__init__(parent)
        self._closed = False
        self._close_requested = False
        self._last_result_rgb_image = None
        self._display_fps = 0.0
        self._display_fps_frames = 0
        self._display_fps_started = time.perf_counter()

        self.setWindowTitle("Sightac Mode 5 - Marker + Hue")
        self.resize(800, 720)

        self.result_image_label = QtWidgets.QLabel()
        self.result_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_image_label.setMinimumSize(320, 240)
        self.result_image_label.setStyleSheet(
            "background: #111; border: 1px solid #444;"
        )

        self.status_label = QtWidgets.QLabel("等待传感器数据...")
        self.status_label.setWordWrap(True)

        self.heatmap_legend = QtWidgets.QLabel(
            "无接触  蓝色                         红色  接触增强"
        )
        self.heatmap_legend.setAlignment(QtCore.Qt.AlignCenter)
        self.heatmap_legend.setMinimumHeight(28)
        self.heatmap_legend.setStyleSheet(
            "QLabel {"
            "color: white; font-weight: bold; padding: 4px;"
            "border: 1px solid #555;"
            "background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0,"
            "stop:0 #000080, stop:0.25 #0000ff,"
            "stop:0.50 #00ffff, stop:0.75 #ffff00, stop:1 #800000"
            ");"
            "}"
        )

        result_group = QtWidgets.QGroupBox("Hue 热力图与标记点位移")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        result_layout.addWidget(self.result_image_label, stretch=1)
        result_layout.addWidget(self.heatmap_legend)

        self.hue_range_spin = QtWidgets.QDoubleSpinBox()
        self.hue_range_spin.setRange(1.0, 180.0)
        self.hue_range_spin.setDecimals(1)
        self.hue_range_spin.setValue(DEFAULT_HUE_MAX)
        self.hue_range_spin.setSuffix(" Hue")

        self.arrow_scale_spin = QtWidgets.QDoubleSpinBox()
        self.arrow_scale_spin.setRange(1.0, 20.0)
        self.arrow_scale_spin.setDecimals(1)
        self.arrow_scale_spin.setValue(DEFAULT_ARROW_SCALE)
        self.arrow_scale_spin.setSuffix("x")

        self.calibrate_button = QtWidgets.QPushButton("重新校准 (B)")
        self.calibrate_button.clicked.connect(self.calibrate_baseline)
        self.calibrate_button.setEnabled(False)

        self.quit_button = QtWidgets.QPushButton("退出")
        self.quit_button.clicked.connect(self.close)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("红色对应 Hue 变化上限"))
        controls.addWidget(self.hue_range_spin)
        controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel("箭头显示倍率"))
        controls.addWidget(self.arrow_scale_spin)
        controls.addStretch(1)
        controls.addWidget(self.calibrate_button)
        controls.addWidget(self.quit_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(result_group, stretch=1)
        layout.addWidget(self.status_label)
        layout.addLayout(controls)

        self.worker = Mode5Worker(
            usb_id=usb_id,
            interval_ms=interval_ms,
            parent=self,
        )
        self.worker.set_visual_settings(
            self.hue_range_spin.value(),
            self.arrow_scale_spin.value(),
        )
        self.worker.frame_available.connect(
            self._on_frame_available,
            QtCore.Qt.QueuedConnection,
        )
        self.worker.initialized.connect(
            self._on_worker_initialized,
            QtCore.Qt.QueuedConnection,
        )
        self.worker.calibration_finished.connect(
            self._on_calibration_finished,
            QtCore.Qt.QueuedConnection,
        )
        self.worker.error_occurred.connect(
            self._on_worker_error,
            QtCore.Qt.QueuedConnection,
        )
        self.worker.finished.connect(self._on_worker_finished)
        self.hue_range_spin.valueChanged.connect(
            self._update_worker_visual_settings
        )
        self.arrow_scale_spin.valueChanged.connect(
            self._update_worker_visual_settings
        )
        self.worker.start()

    def _update_worker_visual_settings(self):
        self.worker.set_visual_settings(
            self.hue_range_spin.value(),
            self.arrow_scale_spin.value(),
        )

    @QtCore.pyqtSlot(str)
    def _on_worker_initialized(self, device_type):
        if self._close_requested:
            return
        self.calibrate_button.setEnabled(True)
        self.status_label.setText(
            f"设备 {device_type} 已连接，等待模式5数据..."
        )

    @QtCore.pyqtSlot()
    def _on_frame_available(self):
        if self._close_requested:
            return

        packet = self.worker.take_latest_packet()
        if packet is None:
            return

        self._last_result_rgb_image = packet["rgb_image"]

        self._refresh_pixmaps()

        self._display_fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._display_fps_started
        if elapsed >= 1.0:
            self._display_fps = self._display_fps_frames / elapsed
            self._display_fps_frames = 0
            self._display_fps_started = now

        self._update_status(packet["stats"])

    def _update_status(self, stats):
        self.status_label.setText(
            f"设备: {stats['device_type']}  |  "
            f"处理 FPS: {stats['pipeline_fps']:.1f}  |  "
            f"显示 FPS: {self._display_fps:.1f}  |  "
            f"耗时: SDK {stats['sensor_ms']:.1f}ms + "
            f"可视化 {stats['visual_ms']:.1f}ms\n"
            f"标记点: {stats['valid_markers']}/{stats['total_markers']}  |  "
            f"最大位移: {stats['max_displacement']:.2f}px  |  "
            f"平均匹配分数: {stats['mean_score']:.3f}\n"
            f"Hue变化强度: max={stats['hue_abs_max']:.2f}, "
            f"sum={stats['hue_abs_sum']:.1f}  |  "
            f"Fz: {stats['fz_total']:.2f}"
        )

    @staticmethod
    def _refresh_pixmap(label, image):
        if image is None:
            return

        height, width = image.shape[:2]
        qimage = QtGui.QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QtGui.QImage.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def _refresh_pixmaps(self):
        self._refresh_pixmap(
            self.result_image_label,
            self._last_result_rgb_image,
        )

    def calibrate_baseline(self):
        if (
            self._closed
            or self._close_requested
            or not self.worker.isRunning()
            or not self.calibrate_button.isEnabled()
        ):
            return

        self.calibrate_button.setEnabled(False)
        self.status_label.setText("正在重新校准，请保持传感器空载...")
        self.worker.request_calibration()

    @QtCore.pyqtSlot(bool, str)
    def _on_calibration_finished(self, success, message):
        if self._close_requested:
            return
        self.calibrate_button.setEnabled(True)
        if success:
            self.status_label.setText(message)
            self._display_fps = 0.0
            self._display_fps_frames = 0
            self._display_fps_started = time.perf_counter()
        else:
            self.status_label.setText(f"校准失败：{message}")
            QtWidgets.QMessageBox.critical(self, "校准失败", message)

    @QtCore.pyqtSlot(str)
    def _on_worker_error(self, message):
        if self._close_requested:
            return
        self.status_label.setText(f"错误：{message}")
        QtWidgets.QMessageBox.critical(self, "模式5运行失败", message)
        self.close()

    @QtCore.pyqtSlot()
    def _on_worker_finished(self):
        if self._close_requested and not self._closed:
            QtCore.QTimer.singleShot(0, self.close)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmaps()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return
        if event.key() == QtCore.Qt.Key_B:
            self.calibrate_baseline()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._closed:
            event.accept()
            return

        self._close_requested = True
        self.calibrate_button.setEnabled(False)
        self.worker.stop()
        if self.worker.isRunning() and not self.worker.wait(1500):
            
            self.status_label.setText("正在停止传感器线程并释放相机...")
            event.ignore()
            return

        self._closed = True
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(
        description="模式5 Hue 热力图和标记点追踪 Qt 可视化"
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
        default=0,
        help="后台采集循环的最小周期；0 表示不额外限速（默认 0）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        [sys.argv[0]]
    )

    viewer = Mode5Viewer(
        usb_id=args.usb_id,
        interval_ms=args.interval_ms,
    )
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

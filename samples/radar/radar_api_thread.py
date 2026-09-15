"""雷达图 Demo：完整依赖 SDK API，并在独立线程中采集完整同帧数据。"""

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QFont


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.algorithm.touch_sensor import TouchSensor
from api.algorithm.sensor_function import SensorFunction
from GUI.radar.radar_features import RadarComputeWorker
from GUI.radar.radar_plot import RadarPlotWidget

@dataclass(frozen=True)
class SensorFramePacket:
    """一次 TouchSensor.update() 完成后生成的同帧数据快照。"""

    frame_id: int
    generation: int
    timestamp: float
    contact_threshold: float
    fx_total: float
    fy_total: float
    fz_total: float
    hue_matrix: np.ndarray
    flow_matrix: np.ndarray
    fx_matrix: np.ndarray
    fy_matrix: np.ndarray
    fz_matrix: np.ndarray


class SensorAcquisitionWorker:
    """独占 TouchSensor，在后台同步执行 SDK 的完整 update pipeline。"""

    def __init__(self, sensor):
        self.sensor = sensor
        self._stop_event = threading.Event()
        self._thread = None
        self._packet_lock = threading.Lock()
        self._latest_packet = None
        self._command_queue = queue.Queue()
        self._event_queue = queue.Queue()
        self._frame_id = 0
        self._generation = 0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="TouchSensorAcquisition",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def request_calibration(self, frames=10):
        self._command_queue.put(("calibrate", int(frames)))

    def take_latest(self):
        """获取最新完整帧；读取后清空，避免重复处理同一个数据包。"""
        with self._packet_lock:
            packet = self._latest_packet
            self._latest_packet = None
        return packet

    def poll_event(self):
        try:
            return self._event_queue.get_nowait()
        except queue.Empty:
            return None

    def _take_command(self):
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    def _run_command(self, command):
        name, value = command
        if name != "calibrate":
            return

        try:
            self.sensor.calibrate_baseline(value)
            self._generation += 1
            with self._packet_lock:
                self._latest_packet = None
            self._event_queue.put(("calibration_done", self._generation))
        except Exception as exc:
            self._event_queue.put(("calibration_error", repr(exc)))

    def _build_packet(self):
        fx_total, fy_total, fz_total = self.sensor.get_total_force()
        hue_matrix = self.sensor.get_hue_matrix()
        flow_matrix = self.sensor.get_flow_matrix()
        fx_matrix, fy_matrix, fz_matrix = self.sensor.get_calib_matrix()

        if (
            fx_total is None
            or fy_total is None
            or fz_total is None
            or hue_matrix is None
            or flow_matrix is None
            or fx_matrix is None
            or fy_matrix is None
            or fz_matrix is None
        ):
            return None

        packet = SensorFramePacket(
            frame_id=self._frame_id,
            generation=self._generation,
            timestamp=time.perf_counter(),
            contact_threshold=float(
                self.sensor.auto_calibration_fz_threshold),
            fx_total=float(fx_total),
            fy_total=float(fy_total),
            fz_total=float(fz_total),
            hue_matrix=hue_matrix.copy(),
            flow_matrix=flow_matrix.copy(),
            fx_matrix=fx_matrix.copy(),
            fy_matrix=fy_matrix.copy(),
            fz_matrix=fz_matrix.copy(),
        )
        self._frame_id += 1
        return packet

    def _loop(self):
        while not self._stop_event.is_set():
            command = self._take_command()
            if command is not None:
                self._run_command(command)
                continue

            try:
                self.sensor.update(mode=0)
                packet = self._build_packet()
                if packet is None:
                    continue
                with self._packet_lock:
                    self._latest_packet = packet
            except Exception as exc:
                self._event_queue.put(("acquisition_error", repr(exc)))


class APIThreadMainWindow(QtWidgets.QWidget):
    """独立雷达界面：Qt 绘制、采集线程和雷达特征线程。"""

    def __init__(self, sensor, sensorfunction, target_fps=30, throttle=1):
        super().__init__()
        self.sensor = sensor
        self.sensorfunction = sensorfunction
        self._current_generation = 0
        self._calibration_pending = False
        self._last_packet_id = -1
        TOUCH_VERSION = self.sensor.get_version()
        FUNCTION_VERSION = self.sensorfunction.get_version()
        self.setWindowTitle(f"Radar Chart — SDK API 同帧采集 Demo Touch VERSION: {TOUCH_VERSION} FUNCTION VERSION: {FUNCTION_VERSION}")
        self.resize(750, 850)
        self.setObjectName("mainWindow")
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.setStyleSheet("""
            QWidget#mainWindow {
                background-color: #0d1117;
                color: #e5e7eb;
                font-family: "Microsoft YaHei UI", "Segoe UI", "PingFang SC";
                font-size: 10.5pt;
            }
            QPushButton {
                color: #ffffff;
                font-weight: 600;
                font-size: 11pt;
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid rgba(255,255,255,90);
                background-color: rgba(255,255,255,45);
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,70);
                border: 1px solid rgba(255,255,255,130);
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._setup_radar_plot()
        layout.addWidget(self.radar_plot, stretch=3)

        dashboard = QtWidgets.QFrame()
        dashboard.setStyleSheet(
            "background: rgba(255,255,255,8); border-radius: 10px;"
            "border: 1px solid rgba(255,255,255,30);")
        dashboard.setFixedHeight(55)
        dashboard_layout = QtWidgets.QHBoxLayout(dashboard)
        dashboard_layout.setContentsMargins(20, 6, 20, 6)
        dashboard_layout.setSpacing(30)
        dashboard_layout.addStretch()

        self.fx_label = QtWidgets.QLabel("Fx\n0.00")
        self.fx_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fx_label.setStyleSheet(
            "color: #62D0FF; font-size: 14pt; font-weight: 700;")
        dashboard_layout.addWidget(self.fx_label)
        dashboard_layout.addWidget(self._make_separator())

        self.fy_label = QtWidgets.QLabel("Fy\n0.00")
        self.fy_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fy_label.setStyleSheet(
            "color: #F3B56B; font-size: 14pt; font-weight: 700;")
        dashboard_layout.addWidget(self.fy_label)
        dashboard_layout.addWidget(self._make_separator())

        self.fz_label = QtWidgets.QLabel("Fz\n0.00")
        self.fz_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fz_label.setStyleSheet(
            "color: #A78BFA; font-size: 14pt; font-weight: 700;")
        dashboard_layout.addWidget(self.fz_label)
        dashboard_layout.addWidget(self._make_separator())

        unit_label = QtWidgets.QLabel("mN")
        unit_label.setAlignment(QtCore.Qt.AlignCenter)
        unit_label.setStyleSheet(
            "color: rgba(255,255,255,100); font-size: 11pt;")
        dashboard_layout.addWidget(unit_label)
        dashboard_layout.addStretch()
        layout.addWidget(dashboard)

        self.diag_label = QtWidgets.QLabel("status: waiting...")
        self.diag_label.setStyleSheet(
            "color: #888; font-size: 10pt; padding: 2px 8px;")
        layout.addWidget(self.diag_label)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(10)

        self.calibrate_btn = QtWidgets.QPushButton("校准 (C)")
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.calibrate_btn.setFixedHeight(30)
        controls.addWidget(self.calibrate_btn)

        self.fps_label = QtWidgets.QLabel("FPS: --")
        self.fps_label.setStyleSheet(
            "color: #62D0FF; font-size: 13pt; font-weight: 700;")
        controls.addWidget(self.fps_label)
        controls.addStretch()

        self.draw_label = QtWidgets.QLabel("draw: --ms")
        self.draw_label.setStyleSheet("color: #aaa; font-size: 11pt;")
        controls.addWidget(self.draw_label)
        controls.addStretch()

        self.throttle_label = QtWidgets.QLabel(f"节流: {throttle}帧/次")
        self.throttle_label.setStyleSheet(
            "color: #F3B56B; font-size: 11pt; font-weight: 600;")
        controls.addWidget(self.throttle_label)
        controls.addStretch()

        hint = QtWidgets.QLabel("T=节流 1/2/3=间隔 C=校准 Q=退出")
        hint.setStyleSheet("color: #666; font-size: 10pt;")
        controls.addWidget(hint)
        layout.addLayout(controls)

        self._radar_worker = RadarComputeWorker()
        self._radar_thread = None
        self._radar_running = False
        self._radar_input = None
        self._radar_input_ready = threading.Event()
        self._radar_output = None
        self._radar_output_lock = threading.Lock()
        self._radar_output_ready = False
        self._start_radar_worker()

        self.target_fps = target_fps
        self.throttle_interval = throttle
        self.frame_counts = 0
        self.fps_start_time = time.time()
        self.throttle_enabled = throttle > 1
        self._updating = False
        self.should_exit = False
        self.button_processing = False

        self.timer = QtCore.QTimer()
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_data)
        update_ms = max(10, int(1000 / max(target_fps, 1)))
        self.timer.start(update_ms)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.sensor_worker = SensorAcquisitionWorker(sensor)
        self.sensor_worker.start()

    @staticmethod
    def _make_separator():
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setStyleSheet(
            "color: rgba(255,255,255,40); max-width: 1px;")
        return separator

    def _setup_radar_plot(self):
        self.radar_plot = RadarPlotWidget(
            RadarComputeWorker.RADAR_LABELS,
            signed_feature_indices=RadarComputeWorker.SIGNED_FEATURE_INDICES,
            signed_feature_deadbands=(
                RadarComputeWorker.SIGNED_FEATURE_DEADBANDS),
        )

    def _start_radar_worker(self):
        if self._radar_thread is not None and self._radar_thread.is_alive():
            return
        self._radar_running = True
        self._radar_thread = threading.Thread(
            target=self._radar_worker_loop,
            name="RadarFeatureWorker",
            daemon=True,
        )
        self._radar_thread.start()

    def _update_fps_display(self):
        now = time.time()
        elapsed = now - self.fps_start_time
        if elapsed > 0.1:
            fps = self.frame_counts / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            if elapsed > 2.0:
                self.frame_counts = 0
                self.fps_start_time = now

    def _radar_worker_loop(self):
        while self._radar_running:
            self._radar_input_ready.wait()
            self._radar_input_ready.clear()
            if not self._radar_running:
                break
            inp = self._radar_input
            if inp is None:
                continue
            try:
                xs, ys, features = self._radar_worker.compute(
                    inp["fz"],
                    inp["hue_matrix"],
                    inp["flow_matrix"],
                    inp["fx_calib"],
                    inp["fy_calib"],
                    inp["fz_calib"],
                    inp["timestamp"],
                    inp["contact_threshold"],
                )
                with self._radar_output_lock:
                    self._radar_output = {
                        "generation": inp["generation"],
                        "frame_id": inp["frame_id"],
                        "xs": xs,
                        "ys": ys,
                        "features": features,
                    }
                    self._radar_output_ready = True
            except Exception as exc:
                print(f"[RadarWorker] 计算错误: {exc}")

    def _try_render_radar(self):
        with self._radar_output_lock:
            if not self._radar_output_ready:
                return None
            output = self._radar_output
            self._radar_output_ready = False

        if output["generation"] != self._current_generation:
            return None

        xs, ys = output["xs"], output["ys"]
        features = output["features"]
        signed_values = {
            index: float(features.get(RadarComputeWorker.FEATURE_NAMES[index], 0.0))
            for index in RadarComputeWorker.SIGNED_FEATURE_INDICES
        }
        self.radar_plot.update_radar(xs, ys, signed_values)
        return features

    def _reset_radar_state(self):
        self._radar_worker = RadarComputeWorker()
        self._radar_input = None
        with self._radar_output_lock:
            self._radar_output = None
            self._radar_output_ready = False

    def _poll_sensor_events(self):
        while True:
            event = self.sensor_worker.poll_event()
            if event is None:
                return
            name, value = event
            if name == "calibration_done":
                self._current_generation = int(value)
                self._last_packet_id = -1
                self._calibration_pending = False
                self._reset_radar_state()
                self.calibrate_btn.setText("校准 (C)")
                self.calibrate_btn.setEnabled(True)
                self.button_processing = False
                self.diag_label.setText("status: SDK 基线校准完成")
                print("Calibration completed!")
            elif name == "calibration_error":
                self._calibration_pending = False
                self.calibrate_btn.setText("校准 (C)")
                self.calibrate_btn.setEnabled(True)
                self.button_processing = False
                self.diag_label.setText("status: SDK 基线校准失败")
                print(f"校准出错: {value}")
            elif name == "acquisition_error":
                self.diag_label.setText("status: SDK 采集错误")
                print(f"[SensorWorker] 数据采集错误: {value}")

    def calibrate(self):
        if self.button_processing or self._calibration_pending:
            return
        self.button_processing = True
        self._calibration_pending = True
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setText("校准中...")
        self.diag_label.setText("status: SDK 基线校准中，请保持空载")
        self.sensor_worker.request_calibration(10)

    def update_data(self):
        if self.should_exit or self._updating:
            return
        self._updating = True
        try:
            self._poll_sensor_events()
            if self._calibration_pending:
                return

            packet = self.sensor_worker.take_latest()
            if packet is None:
                return
            if packet.generation != self._current_generation:
                return
            if packet.frame_id <= self._last_packet_id:
                return
            self._last_packet_id = packet.frame_id

            fx_t = packet.fx_total
            fy_t = packet.fy_total
            fz_t = packet.fz_total

            self.fx_label.setText(f"Fx\n{-fx_t:.2f}")
            self.fy_label.setText(f"Fy\n{-fy_t:.2f}")
            self.fz_label.setText(f"Fz\n{fz_t:.2f}")

            if (
                self.throttle_enabled
                and self.frame_counts % self.throttle_interval != 0
            ):
                self.frame_counts += 1
                if self.frame_counts % 15 == 0:
                    self._update_fps_display()
                return

            t0 = time.perf_counter()
            self._radar_input = {
                "generation": packet.generation,
                "frame_id": packet.frame_id,
                "timestamp": packet.timestamp,
                "contact_threshold": packet.contact_threshold,
                "fx": fx_t,
                "fy": fy_t,
                "fz": fz_t,
                "hue_matrix": packet.hue_matrix,
                "flow_matrix": packet.flow_matrix,
                "fx_calib": packet.fx_matrix,
                "fy_calib": packet.fy_matrix,
                "fz_calib": packet.fz_matrix,
            }
            self._radar_input_ready.set()
            features = self._try_render_radar()
            draw_ms = (time.perf_counter() - t0) * 1000.0
            self.frame_counts += 1

            if self.frame_counts % 5 == 0:
                self.draw_label.setText(f"draw: {draw_ms:.1f}ms")
                self._update_fps_display()

            if features:
                contact = (
                    "接触" if features.get("是否接触", 0.0) > 0.5 else "空载")
                slide = (
                    "滑移" if features.get("是否滑移", 0.0) > 0.5 else "-")
                self.diag_label.setText(
                    f"status: {contact} | slide: {slide} | "
                    f"fz: {fz_t:.1f} mN | frame: {packet.frame_id}")
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"数据处理出错: {exc}")
        finally:
            self._updating = False

    def keyPressEvent(self, event):
        key = event.key()
        if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return
        if key == QtCore.Qt.Key_C:
            self.calibrate()
            return
        if key == QtCore.Qt.Key_T:
            self.throttle_enabled = not self.throttle_enabled
            status = "ON" if self.throttle_enabled else "OFF（每帧）"
            self.throttle_label.setText(
                f"节流: {self.throttle_interval}帧/次 [{status}]")
            return
        if key == QtCore.Qt.Key_1:
            self.throttle_interval = 1
            self.throttle_enabled = False
            self.throttle_label.setText("节流: 关闭（每帧）")
            return
        if key == QtCore.Qt.Key_2:
            self.throttle_interval = 2
            self.throttle_enabled = True
            self.throttle_label.setText("节流: 2帧/次")
            return
        if key == QtCore.Qt.Key_3:
            self.throttle_interval = 3
            self.throttle_enabled = True
            self.throttle_label.setText("节流: 3帧/次")
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.should_exit = True
        self.timer.stop()
        self._radar_running = False
        self._radar_input_ready.set()

        if hasattr(self, "sensor_worker"):
            self.sensor_worker.stop()
        if self._radar_thread is not None:
            self._radar_thread.join(timeout=2.0)
            self._radar_thread = None

        if self.sensor is not None:
            self.sensor.release()
            cap = getattr(self.sensor, "cap", None)
            if cap is not None:
                cap.release()
        event.accept()

def main():
    from GUI.usb_port.usb_port_selector import USBPortSelector

    app = QtWidgets.QApplication(sys.argv)
    port_selector = USBPortSelector()
    if port_selector.exec_() != QtWidgets.QDialog.Accepted:
        print("用户取消了USB端口选择")
        return

    selected_port = port_selector.get_selected_port()
    if selected_port is None:
        print("未选择USB端口")
        return

    print(f"已选择USB端口: {selected_port}")
    sensor = TouchSensor(usb_id=selected_port)
    sensorfunction = SensorFunction()

    print("按 Q/Esc 退出，按 C 校准，按 T/1/2/3 控制节流")
    window = APIThreadMainWindow(sensor, sensorfunction, target_fps=30, throttle=1)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()

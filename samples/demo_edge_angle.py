"""Display edge direction in a Qt interface with a red line."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.algorithm.sensor_function import SensorFunction
from api.algorithm.touch_sensor import TouchSensor


EDGE_THRESHOLD = 1.0
LINE_THICKNESS = 2


def matrix_to_edge_image(matrix, angle, threshold):
    if matrix is None:
        return None

    pressure = np.asarray(matrix, dtype=np.float32)
    grayscale = np.uint8(np.clip(pressure, 0, 255))
    image = np.repeat(grayscale[:, :, None], 3, axis=2)

    if angle is None:
        return np.ascontiguousarray(image)

    mask = pressure > float(threshold)
    points_y, points_x = np.where(mask)
    if len(points_x) < 2:
        return np.ascontiguousarray(image)

    center_x = float(np.mean(points_x))
    center_y = float(np.mean(points_y))
    radians = np.radians(float(angle))
    direction_x = float(np.cos(radians))
    direction_y = float(np.sin(radians))

    height, width = pressure.shape[:2]
    line_length = max(width, height)
    half_length = line_length / 2.0
    start = (
        int(round(center_x - direction_x * half_length)),
        int(round(center_y - direction_y * half_length)),
    )
    end = (
        int(round(center_x + direction_x * half_length)),
        int(round(center_y + direction_y * half_length)),
    )
    cv2.line(image, start, end, (255, 0, 0), LINE_THICKNESS)

    return np.ascontiguousarray(image)


class EdgeAngleViewer(QtWidgets.QWidget):
    def __init__(self, sensor, functions, parent=None):
        super().__init__(parent)
        self.sensor = sensor
        self.functions = functions
        self._closed = False

        self.setWindowTitle("Edge Direction Viewer")
        self.resize(800, 640)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("background: #111; border: 1px solid #444;")

        self.angle_label = QtWidgets.QLabel()
        self.angle_label.setAlignment(QtCore.Qt.AlignCenter)
        self.angle_label.setStyleSheet("font-size: 18px; color: #e8e8e8;")

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.label)
        layout.addWidget(self.angle_label)
        self.setLayout(layout)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        try:
            self.sensor.update(1)
            pressure_matrix = self.sensor.get_hue_matrix()

            angle = self.functions.estimate_edge_angle(
                pressure_matrix,
                threshold=EDGE_THRESHOLD,
            )
            image = matrix_to_edge_image(pressure_matrix, angle, EDGE_THRESHOLD)

            angle_text = "None" if angle is None else f"{angle:.1f} deg"
            self.angle_label.setText(
                f"Angle: {angle_text}    Threshold: {EDGE_THRESHOLD:g}"
            )
            self._set_label_image(image)
        except Exception as exc:
            print(f"Error: {exc}")
            self.close()

    def _set_label_image(self, image):
        if image is None:
            return

        h, w = image.shape[:2]
        qimage = QtGui.QImage(
            image.data,
            w,
            h,
            image.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
            self.label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self.timer.stop()
            self.sensor.release()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    sensor = TouchSensor(usb_id=0)
    functions = SensorFunction()

    viewer = EdgeAngleViewer(sensor, functions)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

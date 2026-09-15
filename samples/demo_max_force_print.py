"""Display the max Hue value in a Qt interface with a red marker."""

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


CELL_SIZE = 1
MARKER_RADIUS = 4


def matrix_to_max_force_image(matrix, max_value, max_row, max_col):
    if matrix is None:
        return None

    pressure = np.asarray(matrix, dtype=np.float32)
    grayscale = np.uint8(np.clip(pressure, 0, 255))
    image = np.repeat(grayscale[:, :, None], 3, axis=2)

    if max_value > 0.0:
        point = (int(max_col), int(max_row))
        cv2.circle(image, point, MARKER_RADIUS, (255, 0, 0), -1)

    return np.ascontiguousarray(image)


class MaxForceViewer(QtWidgets.QWidget):
    def __init__(self, sensor, functions, parent=None):
        super().__init__(parent)
        self.sensor = sensor
        self.functions = functions
        self._closed = False

        self.setWindowTitle("Max Force Viewer")
        self.resize(800, 640)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("background: #111; border: 1px solid #444;")

        self.force_label = QtWidgets.QLabel()
        self.force_label.setAlignment(QtCore.Qt.AlignCenter)
        self.force_label.setStyleSheet("font-size: 18px; color: #e8e8e8;")

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self.label)
        layout.addWidget(self.force_label)
        self.setLayout(layout)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        try:
            self.sensor.update(1)
            pressure_matrix = self.sensor.get_hue_matrix()

            max_value, max_row, max_col = self.functions._compute_max_force_cell(
                pressure_matrix,
                cell_size=CELL_SIZE,
            )
            image = matrix_to_max_force_image(
                pressure_matrix,
                max_value,
                max_row,
                max_col,
            )

            self.force_label.setText(
                f"Max Force: {max_value:.1f}    Cell: ({max_row}, {max_col})"
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

    viewer = MaxForceViewer(sensor, functions)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

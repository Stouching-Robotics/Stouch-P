"""Display the current Hue matrix as a pressure-style image."""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.algorithm.touch_sensor import TouchSensor


def matrix_to_image(matrix):
    if matrix is None:
        return None
    image = np.uint8(np.clip(matrix.astype(np.float32), 0, 255))
    return np.ascontiguousarray(image)


class PressureViewer(QtWidgets.QWidget):
    def __init__(self, sensor, parent=None):
        super().__init__(parent)
        self.sensor = sensor
        self._closed = False

        self.setWindowTitle("Hue Matrix Viewer")
        self.resize(800, 600)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("background: #111; border: 1px solid #444;")

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        try:
            self.sensor.update(1)
            pressure_matrix = self.sensor.get_hue_matrix()
            
            image = matrix_to_image(pressure_matrix)
            self._set_label_image(image)
        except Exception as exc:
            print(f"Error: {exc}")
            self.close()

    def _set_label_image(self, image):
        if image is None:
            return

        h, w = image.shape
        qimage = QtGui.QImage(
            image.data,
            w,
            h,
            image.strides[0],
            QtGui.QImage.Format_Grayscale8,
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

    viewer = PressureViewer(sensor)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

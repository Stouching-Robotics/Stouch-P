"""Display 2D force as a vector arrow in a Qt interface."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import SensorFunction, TouchSensor


class ForcePlaneWidget(QtWidgets.QWidget):
    """绘制平面坐标系，并在中心显示 2D 力向量。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.fx = 0.0
        self.fy = 0.0
        self.magnitude = 0.0
        self.angle = 0.0
        self.max_force_for_scale = 2000.0
        self._min_force = 0.05

    def set_force(self, fx: float, fy: float, magnitude: float, angle: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.magnitude = float(magnitude)
        self.angle = float(angle)
        self.update()

    def set_max_force(self, value: float) -> None:
        self.max_force_for_scale = max(1.0, float(value))
        self.update()

    @staticmethod
    def _draw_arrow_head(painter: QtGui.QPainter, tip: QtCore.QPointF,
                        ux: float, uy: float, length: float = 12.0, width: float = 6.0):
        left = QtCore.QPointF(
            tip.x() - ux * length + uy * width,
            tip.y() - uy * length - ux * width,
        )
        right = QtCore.QPointF(
            tip.x() - ux * length - uy * width,
            tip.y() - uy * length + ux * width,
        )
        painter.drawPolygon(tip, left, right)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect().adjusted(20, 20, -20, -20)
        painter.fillRect(self.rect(), QtGui.QColor("#0f1116"))

        cx = rect.center().x()
        cy = rect.center().y()
        side = min(rect.width(), rect.height())
        plane_half = side * 0.42

        
        plane_rect = QtCore.QRectF(
            cx - plane_half,
            cy - plane_half,
            plane_half * 2,
            plane_half * 2,
        )
        painter.setPen(QtGui.QPen(QtGui.QColor("#5f6270"), 2))
        painter.drawRect(plane_rect)

        
        grid_pen = QtGui.QPen(QtGui.QColor("#2c3140"), 1)
        painter.setPen(grid_pen)
        for i in range(1, 6):
            t = i / 6
            x1 = plane_rect.left() + plane_rect.width() * t
            x2 = plane_rect.top() + plane_rect.height() * t
            painter.drawLine(QtCore.QPointF(x1, plane_rect.top()),
                             QtCore.QPointF(x1, plane_rect.bottom()))
            painter.drawLine(QtCore.QPointF(plane_rect.left(), x2),
                             QtCore.QPointF(plane_rect.right(), x2))

        
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
        painter.drawLine(QtCore.QPointF(cx, plane_rect.top()), QtCore.QPointF(cx, plane_rect.bottom()))
        painter.drawLine(QtCore.QPointF(plane_rect.left(), cy), QtCore.QPointF(plane_rect.right(), cy))

        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setFont(QtGui.QFont("Microsoft YaHei UI", 8))
        painter.setPen(QtGui.QColor("#9aa2b4"))
        painter.drawText(QtCore.QPointF(cx + 4, plane_rect.top() + 12), "+y")
        painter.drawText(QtCore.QPointF(plane_rect.right() - 22, cy - 4), "+x")

        
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawEllipse(QtCore.QPointF(cx, cy), 3.0, 3.0)

        
        painter.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        if self.magnitude <= self._min_force:
            painter.setPen(QtGui.QPen(QtGui.QColor("#9aa2b4"), 2))
            painter.drawText(QtCore.QPointF(plane_rect.left() + 6, plane_rect.bottom() - 8), "No obvious 2D force")
            painter.end()
            return

        scale = plane_half / self.max_force_for_scale
        vx = self.fx
        vy = -self.fy  
        vec_len = math.hypot(vx, vy)
        if vec_len < 1e-9:
            painter.end()
            return

        ux = vx / vec_len
        uy = vy / vec_len
        arrow_len = min(vec_len * scale, plane_half * 0.96)

        
        t = max(0.0, min(self.magnitude / self.max_force_for_scale, 1.0))
        r = int(255 * t)
        b = int(255 * (1 - 0.6 * t))
        g = int(130 * (1 - t))
        color = QtGui.QColor(r, g, b)

        sx = cx
        sy = cy
        ex = cx + ux * arrow_len
        ey = cy + uy * arrow_len

        painter.setPen(QtGui.QPen(color, 3.0))
        painter.setBrush(color)
        painter.drawLine(QtCore.QPointF(sx, sy), QtCore.QPointF(ex, ey))
        self._draw_arrow_head(
            painter,
            QtCore.QPointF(ex, ey),
            ux,
            uy,
            length=12.0 + 8.0 * min(t, 1.0),
            width=5.5 + 2.5 * t,
        )

        
        painter.setPen(QtGui.QColor("#d6deeb"))
        label = f"|Fxy|={self.magnitude:.2f} | angle={self.angle:.2f}°"
        painter.drawText(QtCore.QPointF(plane_rect.left() + 8, plane_rect.top() + 18), label)
        painter.drawText(QtCore.QPointF(plane_rect.left() + 8, plane_rect.bottom() - 8),
                         f"Fx={self.fx:.2f}, Fy={self.fy:.2f}")

        painter.end()


class DirectionDemoWindow(QtWidgets.QWidget):
    def __init__(self, sensor, max_force=2000.0):
        super().__init__()
        self.sensor = sensor
        self.functions = SensorFunction()
        self.setWindowTitle("2D Force Direction (Qt)")
        self.resize(960, 760)

        self.plane_widget = ForcePlaneWidget(self)
        self.plane_widget.set_max_force(max_force)

        self.status_label = QtWidgets.QLabel("State: running")
        self.status_label.setStyleSheet("font-size: 13px; color: #cdd6e8;")
        self.fx_label = QtWidgets.QLabel("Fx: 0.00")
        self.fy_label = QtWidgets.QLabel("Fy: 0.00")
        self.mag_label = QtWidgets.QLabel("Magnitude: 0.00")
        self.ang_label = QtWidgets.QLabel("Angle: 0.00°")

        for lbl in (self.fx_label, self.fy_label, self.mag_label, self.ang_label, self.status_label):
            lbl.setStyleSheet("font-size: 12px; color: #d4def0;")

        self.force_scale_label = QtWidgets.QLabel("Max force for scale: 2000")
        self.force_scale_label.setStyleSheet("font-size: 11px; color: #9ca9c2;")
        self.force_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.force_scale_slider.setRange(200, 10000)
        self.force_scale_slider.setValue(int(max_force))
        self.force_scale_slider.valueChanged.connect(self._on_scale_changed)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(self.force_scale_label)
        scale_row.addWidget(self.force_scale_slider, stretch=1)

        control_panel = QtWidgets.QVBoxLayout()
        control_panel.addWidget(self.status_label)
        control_panel.addWidget(self.fx_label)
        control_panel.addWidget(self.fy_label)
        control_panel.addWidget(self.mag_label)
        control_panel.addWidget(self.ang_label)
        control_panel.addLayout(scale_row)
        control_panel.addStretch(1)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        main_layout.addWidget(self.plane_widget, stretch=3)
        main_layout.addLayout(control_panel, stretch=1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(33)
        self.should_exit = False

    def _on_scale_changed(self, value: int) -> None:
        self.force_scale_label.setText(f"Max force for scale: {value}")
        self.plane_widget.set_max_force(value)

    def update_data(self) -> None:
        if self.should_exit:
            return

        try:
            self.sensor.update(0)
            fx, fy, _ = self.sensor.get_total_force()
            magnitude, angle = self.functions.get_2DForce(fx, fy)
            self.plane_widget.set_force(fx, fy, magnitude, angle)

            self.fx_label.setText(f"Fx: {fx:.2f} mN")
            self.fy_label.setText(f"Fy: {fy:.2f} mN")
            self.mag_label.setText(f"Magnitude: {magnitude:.2f} mN")
            self.ang_label.setText(f"Angle: {angle:.2f}°")
        except Exception as exc:
            self.status_label.setText(f"Error: {exc}")
            self.should_exit = True
            self.timer.stop()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.should_exit = True
        self.timer.stop()
        self.sensor.release()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            return
        super().keyPressEvent(event)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    sensor = TouchSensor(usb_id=0)
    window = DirectionDemoWindow(sensor, max_force=2000.0)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

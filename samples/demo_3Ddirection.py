"""Display 3D force as a draggable OpenGL vector in a PyQt interface."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import SensorFunction, TouchSensor

pg.setConfigOptions(antialias=True)


class Force3DGLWidget(gl.GLViewWidget):
    """OpenGL 3D 力向量显示器（可拖动转动）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor((0, 0, 0, 255))
        self.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=650, elevation=38, azimuth=-54)

        self.axis_len = 260.0
        self.max_force_for_scale = 2000.0
        self._min_force = 0.05

        self.axis_items = []
        self._build_axis_and_grid()
        self._build_arrow_primitives()
        self._reset_force_visual()

        
        self.setMouseTracking(True)

    def _build_axis_and_grid(self):
        half = float(self.axis_len)

        
        
        self.xy_plane = self._make_xy_plane(half)
        self.addItem(self.xy_plane)

        self.xy_grid = self._make_xy_grid(half, spacing=max(20.0, half / 9.0))
        self.xy_grid.setGLOptions('opaque')
        self.addItem(self.xy_grid)

        self.xy_border = self._make_xy_border(half)
        self.xy_border.setGLOptions('opaque')
        self.addItem(self.xy_border)

        axis_white = (1.0, 1.0, 1.0, 1.0)
        
        self.axis_items.append(self._make_axis((0, 0, 0), (self.axis_len, 0, 0), axis_white))  
        self.axis_items.append(self._make_axis((0, 0, 0), (0, self.axis_len, 0), axis_white))  
        self.axis_items.append(self._make_axis((0, 0, 0), (0, 0, self.axis_len), axis_white))  
        self.axis_items.append(self._make_axis((0, 0, 0), (-self.axis_len, 0, 0), axis_white))
        self.axis_items.append(self._make_axis((0, 0, 0), (0, -self.axis_len, 0), axis_white))
        self.axis_items.append(self._make_axis((0, 0, 0), (0, 0, -self.axis_len), axis_white))

        for item in self.axis_items:
            item.setGLOptions('opaque')
            self.addItem(item)

        self._axis_label_items = [
            ("X", (self.axis_len * 1.06, 0.0, 0.0), axis_white),
            ("Y", (0.0, self.axis_len * 1.06, 0.0), axis_white),
            ("Z", (0.0, 0.0, self.axis_len * 1.06), axis_white),
        ]

    @staticmethod
    def _make_xy_plane(half: float):
        z = -1.2
        verts = np.array([
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0],
            [3, 2, 0],
        ], dtype=np.uint32)
        item = gl.GLMeshItem(
            meshdata=gl.MeshData(vertexes=verts, faces=faces),
            color=(1.0, 1.0, 1.0, 0.18),
            shader='shaded',
            drawEdges=False,
            smooth=False,
        )
        item.setGLOptions('translucent')
        return item

    @staticmethod
    def _make_xy_grid(half: float, spacing: float):
        z = -0.6
        pts = []
        x = -half
        while x <= half + 1e-6:
            pts.append((x, -half, z))
            pts.append((x, half, z))
            x += spacing
        y = -half
        while y <= half + 1e-6:
            pts.append((-half, y, z))
            pts.append((half, y, z))
            y += spacing
        return gl.GLLinePlotItem(
            pos=np.asarray(pts, dtype=np.float32),
            color=(1.0, 1.0, 1.0, 0.75),
            width=1.5,
            antialias=True,
            mode='lines'
        )

    @staticmethod
    def _make_xy_border(half: float):
        z = -0.2
        pts = np.array([
            [-half, -half, z],
            [half, -half, z],
            [half, -half, z],
            [half, half, z],
            [half, half, z],
            [-half, half, z],
            [-half, half, z],
            [-half, -half, z],
        ], dtype=np.float32)
        return gl.GLLinePlotItem(
            pos=pts,
            color=(1.0, 1.0, 1.0, 1.0),
            width=2.5,
            antialias=True,
            mode='lines'
        )

    @staticmethod
    def _make_axis_static(start, end, color):
        return gl.GLLinePlotItem(
            pos=np.array([start, end], dtype=np.float32),
            color=color,
            width=1.1,
            antialias=True,
            mode='lines'
        )

    @staticmethod
    def _qmatrix_to_numpy(m):
        return np.array([
            [m.row(0).x(), m.row(0).y(), m.row(0).z(), m.row(0).w()],
            [m.row(1).x(), m.row(1).y(), m.row(1).z(), m.row(1).w()],
            [m.row(2).x(), m.row(2).y(), m.row(2).z(), m.row(2).w()],
            [m.row(3).x(), m.row(3).y(), m.row(3).z(), m.row(3).w()],
        ], dtype=np.float32)

    def _project_world_to_screen(self, p_world):
        view_m = self._qmatrix_to_numpy(self.viewMatrix())
        proj_m = self._qmatrix_to_numpy(self.projectionMatrix())
        v = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32)
        clip = proj_m @ (view_m @ v)
        w = float(clip[3])
        if abs(w) < 1e-6:
            return None
        ndc = clip[:3] / w
        x = (float(ndc[0]) * 0.5 + 0.5) * self.width()
        y = (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * self.height()
        return QtCore.QPointF(x, y)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
        for name, p_world, color in self._axis_label_items:
            pt = self._project_world_to_screen(p_world)
            if pt is None:
                continue
            if 0 <= pt.x() <= self.width() and 0 <= pt.y() <= self.height():
                painter.setPen(QtGui.QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
                painter.drawText(pt, name)
        painter.end()

    @staticmethod
    def _make_axis(start, end, color):
        return gl.GLLinePlotItem(
            pos=np.array([start, end], dtype=np.float32),
            color=color,
            width=4.0,
            antialias=True,
            mode='lines'
        )

    def _build_arrow_primitives(self):
        self.arrow_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.95, 0.67, 0.12, 1.0),
            width=2.5,
            antialias=True,
            mode='lines'
        )
        self.arrow_line.setGLOptions('opaque')
        self.addItem(self.arrow_line)

        head_mesh = gl.MeshData.cylinder(rows=2, cols=16, radius=[3.8, 0.0], length=12.0)
        self.arrow_head = gl.GLMeshItem(
            meshdata=head_mesh,
            color=(0.95, 0.67, 0.12, 0.98),
            shader='shaded',
            drawEdges=False,
            smooth=True,
        )
        self.arrow_head.setGLOptions('translucent')
        self.addItem(self.arrow_head)

        self.origin_point = gl.GLScatterPlotItem(
            pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            color=(0.96, 0.98, 1.0, 1.0),
            size=5.5,
            pxMode=False,
        )
        self.origin_point.setGLOptions('opaque')
        self.addItem(self.origin_point)

        self._arrow_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def set_max_force(self, value: float) -> None:
        self.max_force_for_scale = max(1.0, float(value))

    def _force_color(self, t: float):
        t = float(np.clip(t, 0.0, 1.0))
        r = 0.16 + 0.84 * t
        g = 0.85 * (1.0 - t)
        b = 1.0 - 0.6 * t
        return float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1)), 1.0

    def _rotate_head_to_vector(self, direction):
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return pg.Transform3D()
        direction = direction / norm

        rot = pg.Transform3D()
        if np.allclose(direction, z_axis):
            return rot
        if np.allclose(direction, -z_axis):
            rot.rotate(180, 1, 0, 0)
            return rot

        axis = np.cross(z_axis, direction)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return rot
        axis = axis / axis_norm
        angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
        rot.rotate(angle, axis[0], axis[1], axis[2])
        return rot

    def set_force(self, fx: float, fy: float, fz: float) -> None:
        fx = float(fx)
        fy = float(fy)
        fz = float(fz)
        raw = np.array([fx, fy, fz], dtype=np.float32)
        raw_len = float(np.linalg.norm(raw))

        if raw_len < self._min_force:
            draw_vec = np.array([0.0, 0.0, self.axis_len * 0.08], dtype=np.float32)
            normalized_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            display_mag = 0.0
        else:
            scale = min(raw_len, self.max_force_for_scale) / self.max_force_for_scale
            draw_vec = raw / raw_len * (scale * self.axis_len)
            normalized_dir = draw_vec / (np.linalg.norm(draw_vec) + 1e-9)
            display_mag = raw_len

        t = display_mag / self.max_force_for_scale if self.max_force_for_scale > 0 else 0.0
        color = self._force_color(t)

        line_pos = np.array([[0.0, 0.0, 0.0], draw_vec], dtype=np.float32)
        line_w = max(1.2, 1.8 + 4.0 * min(t, 1.0))
        self.arrow_line.setData(
            pos=line_pos,
            color=color,
            width=line_w,
            mode='lines'
        )

        
        head_len = 10.0 + 10.0 * min(t, 1.0)
        head_half = head_len * 0.5
        base_offset = normalized_dir * head_half
        head_origin = draw_vec - base_offset

        tr = pg.Transform3D()
        tr.translate(float(head_origin[0]), float(head_origin[1]), float(head_origin[2]))
        tr = tr * self._rotate_head_to_vector(normalized_dir)
        tr.scale(0.12 + 0.45 * min(t, 1.0), 0.12 + 0.45 * min(t, 1.0), head_len)
        self.arrow_head.setTransform(tr)
        self.arrow_head.setColor(color)

        self._arrow_vec = raw

    def _reset_force_visual(self):
        self.set_force(0.0, 0.0, 0.0)

    def reset_view(self):
        self.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=650, elevation=38, azimuth=-54)


class Direction3DOpenGLWindow(QtWidgets.QWidget):
    def __init__(self, sensor, max_force=2000.0):
        super().__init__()
        self.sensor = sensor
        self.functions = SensorFunction()
        self.setWindowTitle("3D Force Direction (OpenGL)")
        self.resize(1120, 780)

        self.canvas = Force3DGLWidget(self)
        self.canvas.set_max_force(max_force)

        self.status_label = QtWidgets.QLabel("State: running")
        self.status_label.setStyleSheet("font-size: 13px; color: #d0d7e2;")

        self.fx_label = QtWidgets.QLabel("Fx: 0.00")
        self.fy_label = QtWidgets.QLabel("Fy: 0.00")
        self.fz_label = QtWidgets.QLabel("Fz: 0.00")
        self.mag_label = QtWidgets.QLabel("Magnitude: 0.00")
        self.az_label = QtWidgets.QLabel("Azimuth: 0.00°")
        self.pitch_label = QtWidgets.QLabel("Pitch: 0.00°")
        for lbl in (self.fx_label, self.fy_label, self.fz_label,
                    self.mag_label, self.az_label, self.pitch_label, self.status_label):
            lbl.setStyleSheet("font-size: 12px; color: #d8e0ef;")

        self.help_label = QtWidgets.QLabel("Tip: Drag with mouse to rotate 3D view, wheel to zoom")
        self.help_label.setStyleSheet("font-size: 10px; color: #9ca9c2;")

        self.force_scale_label = QtWidgets.QLabel("Max force for scale: 2000")
        self.force_scale_label.setStyleSheet("font-size: 11px; color: #9ca9c2;")
        self.force_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.force_scale_slider.setRange(200, 12000)
        self.force_scale_slider.setValue(int(max_force))
        self.force_scale_slider.valueChanged.connect(self._on_scale_changed)

        self.reset_btn = QtWidgets.QPushButton("Reset View")
        self.reset_btn.setFixedHeight(30)
        self.reset_btn.clicked.connect(self.canvas.reset_view)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(self.force_scale_label)
        scale_row.addWidget(self.force_scale_slider, stretch=1)

        control = QtWidgets.QVBoxLayout()
        control.addWidget(self.status_label)
        control.addWidget(self.help_label)
        control.addWidget(self.fx_label)
        control.addWidget(self.fy_label)
        control.addWidget(self.fz_label)
        control.addWidget(self.mag_label)
        control.addWidget(self.az_label)
        control.addWidget(self.pitch_label)
        control.addWidget(self.reset_btn)
        control.addLayout(scale_row)
        control.addStretch(1)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self.canvas, stretch=3)
        root.addLayout(control, stretch=1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(33)
        self.should_exit = False

    def _on_scale_changed(self, value: int) -> None:
        self.force_scale_label.setText(f"Max force for scale: {value}")
        self.canvas.set_max_force(value)

    def update_data(self) -> None:
        if self.should_exit:
            return
        try:
            self.sensor.update(0)
            fx, fy, fz = self.sensor.get_total_force()
            magnitude, azimuth, pitch = self.functions.get_3DForce(fx, fy, fz)

            self.canvas.set_force(fx, fy, fz)
            self.fx_label.setText(f"Fx: {fx:.2f} mN")
            self.fy_label.setText(f"Fy: {fy:.2f} mN")
            self.fz_label.setText(f"Fz: {fz:.2f} mN")
            self.mag_label.setText(f"|F|: {magnitude:.2f} mN")
            self.az_label.setText(f"Azimuth: {azimuth:.2f}°")
            self.pitch_label.setText(f"Pitch: {pitch:.2f}°")
            self.status_label.setText("State: running")
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
        if event.key() == QtCore.Qt.Key_R:
            self.canvas.reset_view()
            return
        super().keyPressEvent(event)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    sensor = TouchSensor(usb_id=0)
    win = Direction3DOpenGLWindow(sensor, max_force=2000.0)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

"""雷达图绘制组件：只负责 Qt/pyqtgraph 图形显示。"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont


pg.setConfigOptions(antialias=True)


class RadarPlotWidget(pg.PlotWidget):
    """绘制固定雷达网格，并用外部传入的坐标更新数据多边形。"""

    def __init__(
        self,
        feature_names,
        signed_feature_indices=(),
        signed_feature_deadbands=None,
        parent=None,
        label_font_size=8,
    ):
        super().__init__(parent=parent)
        self.feature_names = tuple(feature_names)
        self.signed_feature_indices = tuple(signed_feature_indices)
        self.signed_feature_deadbands = dict(
            signed_feature_deadbands or {})
        self.label_font_size = int(label_font_size)
        self._signed_markers = {}
        self._setup_plot()

    def _setup_plot(self):
        self.setBackground((0, 0, 0, 0))
        self.setStyleSheet("background: transparent; border: none;")
        self.setMinimumHeight(360)
        self.setAspectLocked(True)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setXRange(-1.22, 1.22)
        self.setYRange(-1.22, 1.22)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)

        angles = np.linspace(
            0,
            2 * np.pi,
            len(self.feature_names),
            endpoint=False,
            dtype=np.float64,
        )
        theta = np.linspace(0, 2 * np.pi, 256, dtype=np.float64)

        circle_pen = pg.mkPen("#555555", width=0.8)
        for radius in (0.2, 0.4, 0.6, 0.8, 1.0):
            self.plot(
                radius * np.cos(theta),
                radius * np.sin(theta),
                pen=circle_pen,
            )

        line_pen = pg.mkPen("#555555", width=0.5)
        for angle in angles:
            draw_angle = np.pi / 2 - angle
            self.plot(
                [0, np.cos(draw_angle)],
                [0, np.sin(draw_angle)],
                pen=line_pen,
            )

        for name, angle in zip(self.feature_names, angles):
            draw_angle = np.pi / 2 - angle
            label = pg.TextItem(name, color="#e5e7eb", anchor=(0.5, 0.5))
            label.setFont(QFont("Microsoft YaHei UI", self.label_font_size))
            self.addItem(label)
            label.setPos(
                1.16 * np.cos(draw_angle),
                1.16 * np.sin(draw_angle),
            )

        tick_angle = np.pi / 4
        for radius in (0.2, 0.4, 0.6, 0.8, 1.0):
            tick = pg.TextItem(
                f"{radius:.1f}",
                color="#888888",
                anchor=(0.5, 0.5),
            )
            tick.setFont(QFont("Microsoft YaHei UI", 7))
            self.addItem(tick)
            tick.setPos(
                radius * np.cos(tick_angle),
                radius * np.sin(tick_angle),
            )

        self._fill_item = QtWidgets.QGraphicsPathItem()
        self._fill_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self._fill_item.setBrush(
            QtGui.QBrush(QtGui.QColor(98, 208, 255, 80)))
        self.addItem(self._fill_item)

        self._line_item = pg.PlotDataItem(
            [],
            [],
            pen=pg.mkPen("#62D0FF", width=1.8),
        )
        self._line_item.setZValue(1)
        self.addItem(self._line_item)

        for index in self.signed_feature_indices:
            marker = pg.ScatterPlotItem(
                [0.0],
                [0.0],
                size=11,
                symbol="o",
                brush=pg.mkBrush("#888888"),
                pen=pg.mkPen("#e5e7eb", width=0.8),
            )
            marker.setZValue(2)
            self.addItem(marker)
            self._signed_markers[index] = marker

    def update_radar(self, xs, ys, signed_values=None):
        """用闭合的笛卡尔坐标更新雷达折线和填充区域。"""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if xs.size == 0 or xs.shape != ys.shape:
            return

        self._line_item.setData(xs, ys)
        path = QtGui.QPainterPath()
        path.moveTo(float(xs[0]), float(ys[0]))
        for px, py in zip(xs[1:], ys[1:]):
            path.lineTo(float(px), float(py))
        path.closeSubpath()
        self._fill_item.setPath(path)

        signed_values = signed_values or {}
        for index, marker in self._signed_markers.items():
            value = float(signed_values.get(index, 0.0))
            deadband = max(
                float(self.signed_feature_deadbands.get(index, 0.0)),
                0.0,
            )
            if abs(value) <= deadband:
                color = "#888888"  
            elif value > 0.0:
                color = "#62D0FF"  
            else:
                color = "#F3B56B"  
            marker.setData(
                [float(xs[index])],
                [float(ys[index])],
                size=11,
                brush=pg.mkBrush(color),
                pen=pg.mkPen("#e5e7eb", width=0.8),
            )

    def clear_radar(self):
        """清空动态数据，保留固定网格和标签。"""
        self._line_item.setData([], [])
        self._fill_item.setPath(QtGui.QPainterPath())
        for marker in self._signed_markers.values():
            marker.setData([], [])

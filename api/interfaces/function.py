"""SensorFunction 公开接口"""

from ..algorithm.sensor_function import SensorFunction as _Engine


class SensorFunction:
    """视触觉传感器后处理公开接口"""

    def __init__(self, touch_thresholds=None):
        self._engine = _Engine(touch_thresholds)

    def get_2DForce(self, fx_total, fy_total):
        return self._engine.get_2DForce(fx_total, fy_total)

    def get_3DForce(self, fx_total, fy_total, fz_total):
        return self._engine.get_3DForce(fx_total, fy_total, fz_total)

    def get_COF(self, fx_total, fy_total, fz_total):
        return self._engine.get_COF(fx_total, fy_total, fz_total)

    def detect_touch_status(self, *args, **kwargs):
        return self._engine.detect_touch_status(*args, **kwargs)

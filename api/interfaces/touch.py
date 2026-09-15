"""TouchSensor 公开接口"""

from ..algorithm.touch_sensor import TouchSensor as _Engine


class TouchSensor:
    """触觉传感器公开接口"""

    def __init__(self, usb_id: int):
        self._engine = _Engine(usb_id)

    def get_device_info(self):
        if self._engine.xu_camera and self._engine.xu_camera.is_available():
            return self._engine.xu_camera.get_device_info()
        return None

    def update(self, mode):
        return self._engine.update(mode)

    def release(self):
        return self._engine.release()

    def search_ports(self):
        return self._engine.search_ports()

    def get_total_data(self):
        return self._engine.get_total_data()

    def get_total_force(self):
        return self._engine.get_total_force()

    def get_calib_matrix(self):
        return self._engine.fx_matrix, self._engine.fy_matrix, self._engine.fz_matrix

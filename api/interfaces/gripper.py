"""Gripper 夹爪公开接口"""

from ..algorithm.sensor_gripper import Gripper as _Engine


class Gripper:
    """夹爪双传感器公开接口"""

    def __init__(self, usb_ids, mode=0):
        self._engine = _Engine(usb_ids, mode)

    def calibrate(self, calibration_frames=5):
        return self._engine.calibrate(calibration_frames)

    def update(self, mode=None):
        return self._engine.update(mode)

    def get_forces(self):
        return self._engine.get_forces()

    def close(self):
        return self._engine.close()

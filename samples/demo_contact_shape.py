"""Classify the largest Hue contact region shape."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.algorithm.sensor_function import SensorFunction
from api.algorithm.touch_sensor import TouchSensor


def main():
    sensor = TouchSensor(usb_id=0)
    functions = SensorFunction()

    try:
        while True:
            
            sensor.update_frame()
            sensor.update_calculation(1)
            pressure_matrix = sensor.get_hue_matrix()

            shape, _, bbox = functions.detect_contact_shape(
                pressure_matrix,
                threshold=1.0,
            )
            print(f"shape: {shape}, bbox: {bbox}")
    except KeyboardInterrupt:
        pass
    finally:
        sensor.release()


if __name__ == "__main__":
    main()

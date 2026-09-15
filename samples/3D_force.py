"""3D 力值实时打印 Demo — 实时输出法向力 Fz 和切向力 Fx、Fy。

Usage:
    python samples/3D_force.py            # 自动扫描传感器端口
    python samples/3D_force.py <usb_id>   # 手动指定 USB 端口

退出:
    Ctrl + C
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.algorithm.touch_sensor import TouchSensor
from api.hardware.xu_camera import XuCamera


def auto_scan_port(max_devices: int = 4) -> int:
    """自动扫描 Sonix 传感器端口（VID/PID 匹配，不打开视频流）。

    找到设备时返回第一个端口的 OpenCV 索引；未找到时回退到默认端口 0。
    """
    ports = XuCamera.enumerate_cv_indices(max_devices)
    if ports:
        print(f"自动扫描到 {len(ports)} 个传感器端口: {ports}，使用 {ports[0]}")
        return ports[0]
    print("自动扫描未找到 Sonix 传感器，回退到默认端口 0")
    return 0


def main(usb_id: int | None = None):
    if usb_id is None:
        usb_id = auto_scan_port()

    sensor = TouchSensor(usb_id=usb_id)
    print(f"传感器已连接, device_type={sensor.device_type}")
    print("实时输出 Fx / Fy / Fz (mN)，按 Ctrl+C 退出\n")

    frame_count = 0
    t0 = time.perf_counter()

    try:
        while True:
            
            sensor.update(mode=0)

            
            fx, fy, fz = sensor.get_total_force()
            frame_count += 1

            elapsed = time.perf_counter() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            
            print(f"\rFx={fx:6d} mN   Fy={fy:6d} mN   Fz={fz:6d} mN   "
                  f"|  frame#{frame_count}   {fps:4.1f} FPS   ",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as exc:
        print(f"\n错误: {exc}")
        traceback.print_exc()
    finally:
        sensor.release()
        print("传感器已释放")


if __name__ == "__main__":
    usb_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(usb_id)

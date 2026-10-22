# SighTac API 接口参考文档

> 版本: 3.4.0 | 日期: 2026-10-22

---

## TouchSensor

单路触觉传感器。构造时自动初始化相机、读取 Flash 标定参数、执行基线校准。

**接口文件：** `api/interfaces/touch.py`

**导入：** `from api import TouchSensor`

**构造：** `sensor = TouchSensor(usb_id=0)`

| 方法 | 说明 |
|------|------|
| `update(mode)` | — `mode`: 0, 1, 2, 5|
| `get_device_info()` | return `dict` 或 `None` |
| `search_ports()` | return `list[int]` |
| `get_total_data()` | return `(float, float, float)` |
| `get_total_force()` | return `(int, int, int)` |
| `get_calib_matrix()` | return `(ndarray, ndarray, ndarray)` |
| `release()` | — |

**典型用法：**
```python
from api import TouchSensor

sensor = TouchSensor(usb_id=0)
while True:
    sensor.update(mode=0)
    fx, fy, fz = sensor.get_total_data()
    print(f"Fx={fx:.2f}, Fy={fy:.2f}, Fz={fz:.2f}")
sensor.release()
```

---

## SensorFunction

视触觉传感器后处理工具集。纯计算，不访问硬件。

**接口文件：** `api/interfaces/function.py`

**导入：** `from api import SensorFunction`

**构造：** `sf = SensorFunction()`

| 方法 | 说明 |
|------|------|
| `get_2DForce(fx, fy)` | return `(magnitude, angle_deg)` |
| `get_3DForce(fx, fy, fz)` | return `(magnitude, azimuth_deg, pitch_deg)` |
| `get_COF(fx, fy, fz)` | return `float` |
| `detect_touch_status(fz_total, hue_matrix_filtered, thresholds=None, history=None, flow_matrix_filtered=None)` | return `(status, history)` |

---

## Gripper

双传感器夹爪。内部持有左右两个 TouchSensor，提供同步更新和汇总输出。

**接口文件：** `api/interfaces/gripper.py`

**导入：** `from api import Gripper`

**构造：** `gripper = Gripper({"left": 0, "right": 1}, mode=0)`

| 方法 | 说明 |
|------|------|
| `calibrate(calibration_frames=5)` | — |
| `update(mode=None)` | — |
| `get_forces()` | return `{"left": (fx,fy,fz), "right": (fx,fy,fz)}` |
| `close()` | — |


**典型用法：**
```python
from api import Gripper

gripper = Gripper({"left": 0, "right": 1})
gripper.update()
left_fx, left_fy, left_fz = gripper.get_forces()["left"]
gripper.close()
```

---

## SensorHand

五指灵巧手封装。内部固定持有 thumb/index/middle/ring/little 五个 TouchSensor。

**接口文件：** `api/interfaces/hand.py`

**导入：** `from api.interfaces import SensorHand`

**构造：** `hand = SensorHand({"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}, mode=0)`

| 方法 | 说明 |
|------|------|
| `calibrate(calibration_frames=5)` | — |
| `update(mode=None)` | — |
| `get_forces()` | return `dict` |
| `close()` | — |

**典型用法：**
```python
from api.interfaces import SensorHand

hand = SensorHand({"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4})
hand.update()
thumb_fx, thumb_fy, thumb_fz = hand.get_forces()["thumb"]
hand.close()
```

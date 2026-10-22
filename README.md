# 视塔科技视触觉传感器 SDK 

<div align="center">

[![Version](https://img.shields.io/badge/Version-3.4.0-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV_headless-4.x-green.svg)](https://opencv.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

##  项目简介

该传感器是一个视触觉传感器，通过USB摄像头实时检测并且可视化**压力分布**和**切向力**。该系统利用**光学流算法**（Optical Flow）和**色相变化分析**（HSV Hue Shift）来计算触摸表面的力学特性，并提供专业的3D可视化界面。

### 核心特性

- **法向压力检测** — 基于 HSV 色相偏移计算压力分布矩阵
- **切向力场检测** — 使用 Farneback 稠密光流算法计算切向力矢量
- **力方向分析** — 二维/三维力模长、方向角、俯仰角、摩擦系数计算
- **自动校准** — 连续低压力窗口自动重设基线，无需手动干预；支持空载异常检测与报告
- **阈值滤波** — Hue 差值阈值滤波 + 光流曼哈顿距离阈值滤波 + 低饱和度噪声抑制；曲面款/斜面款额外支持 R/B 通道滤波
- **空间滤波** — 曲面款、斜面款支持空间滤波 + 阈值滤波组合去噪
- **多形态设备支持** — 曲面款（curved）、斜面款（bevel）两种传感器形态，自动识别并适配对应的标定与渲染模式
- **分区/分段力标定** — 曲面款全局标定；斜面款三点空间分区标定
- **触摸状态判断** — 基于法向力与连续帧 hue 光流的 idle/contact/slide 状态机
- **接触特征提取** — 接触面积、质心、形态、棱角度估计、力矩计算
- **硬件标定支持** — 从摄像头 Serial Flash 读取力标定系数和 ROI 参数
- **XU 摄像头控制** — 通过 Sonix SDK 读写相机参数（Windows / Linux）
- **3D 可视化 GUI** — 实时压力表面、力场箭头、力历史曲线；支持指尖/平面模式切换
- **参数动态配置** — GUI 内实时调整光流、网格、缩放等参数
- **夹爪双传感器封装** — Gripper 类统一管理左右两个传感器，支持坐标系可视化
- **灵巧手多传感器封装** — SensorHand 类统一管理五指传感器
- **跨平台支持** — Windows / Linux（macOS 仅支持视频采集，不支持 XU 控制与 Flash 标定）
- **USB 端口自动扫描** — 智能检测可用摄像头设备

---

##  环境部署

### 环境要求

- **Python**: 3.10 或更高版本
- **操作系统**: Windows / Linux
- **硬件**: USB摄像头（用于触觉传感）

### 安装步骤

#### 1. 配置环境
#### 
```bash
# 安装系统级依赖
sudo apt-get update
# 安装 OpenGL 和系统库支持
sudo apt-get install python3-venv libgl1 libegl1 libxkbcommon-x11-0
# 创建工作目录
cd sightac_sdk
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**依赖列表**：
```
opencv-python-headless        # 计算机视觉和光流算法（headless 防止与 PyQt5 冲突）
numpy==2.0.2                  # 数值计算
PyQt5==5.15.11                # GUI框架
pyqtgraph==0.13.7             # 科学绘图和3D可视化
matplotlib==3.9.4             # 数据绘图
PyOpenGL==3.1.10              # OpenGL 3D渲染
PyOpenGL-accelerate==3.1.10   # OpenGL加速
numpy-stl==3.2.0              # STL 3D模型加载
pillow==11.3.0                # 图像处理
pyyaml==6.0.2                 # 配置文件加载
```

#### 2. 连接USB摄像头

确保USB摄像头已连接到电脑，系统可以识别设备。

#### 3. 运行主程序

```bash
# 从项目根目录运行
cd sightac_sdk
python GUI/demo_gui.py
```

---

## 使用指南

### 主应用程序 (GUI/demo_gui.py)

#### 启动流程

1. **运行程序**
   ```bash
   python GUI/demo_gui.py
   ```

2. **USB端口选择**
   - 程序启动后会自动选择USB端口并连接进入主界面
   - 也可以“更换设备”手动扫描可用的USB摄像头（端口0-9）
   - 显示每个端口的分辨率和帧率信息
   - 可预览摄像头画面
   - 选择目标端口后点击"进入主界面"

3. **传感器初始化**
   - 自动检测有效区域（ROI）
   - 校准基准矩阵（采集5帧静态背景）
   - 初始化3D可视化窗口

4. **实时监测**
   - 3D压力表面：显示实时压力分布
   - 热力图：压力分布热力图
   - 切向力场：显示光流矢量箭头
   - 力历史曲线：显示最近3秒的Fx、Fy、Fz数据


---

### 示例程序

项目提供多个示例程序，用于快速测试和学习API用法。

#### 示例1: 三维力矢量显示 (demo_3Ddirection)

可拖拽旋转的 OpenGL 三维力矢量实时显示。

```bash
python samples/demo_3Ddirection.py
```

#### 示例2: 二维力方向 (demo_2Ddirection)

实时打印二维力模长与方向角。

```bash
python samples/demo_2Ddirection.py
```

#### 示例3: 接触形状识别 (demo_contact_shape)

识别最大 Hue 接触区域的形状并分类（Qt 界面可视化）。

```bash
python samples/demo_contact_shape.py
```

#### 示例4: 接触重心显示 (demo_center_of_gravity)

Qt 界面实时显示接触重心位置，以红色标记标出。

```bash
python samples/demo_center_of_gravity.py
```

---

##  核心API使用

### TouchSensor — 单传感器核心类

`TouchSensor` 是单个视触觉传感器的核心 API，负责 USB 摄像头初始化、Flash 标定参数读取、ROI/旋转预处理、基线校准、Hue/光流矩阵计算、三轴力标定与输出。支持曲面（curved）和斜面（bevel）两种设备类型，具备自动校准能力。

#### 初始化

```python
from api import TouchSensor

sensor = TouchSensor(usb_id=0)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `usb_id` | `int` | USB 摄像头设备 ID，通常为 `0`, `1`, `2` 等整数。构造时自动完成相机初始化和基线校准。 |


#### 主要方法

##### 1. 主循环更新

```python
sensor.update(mode=0)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `mode` | `int` | `0` = Hue + 光流，`1` = 仅 Hue，`2` = 仅光流，`5` = 标记点模板追踪 |

##### 2. 获取三轴总力

```python
fx, fy, fz = sensor.get_total_force()     # 最终三轴总力（去噪后整数，最常用）
fx, fy, fz = sensor.get_total_data()      # 原始浮点三轴力（未钳位）
```

##### 3. 获取逐像素力标定矩阵

```python
fx_mat, fy_mat, fz_mat = sensor.get_calib_matrix()
# 返回: (np.ndarray | None, np.ndarray | None, np.ndarray | None)
#       三轴逐像素力标定矩阵
```

##### 4. 获取设备信息

```python
info = sensor.get_device_info()
# 返回: dict | None
# 字段: chip_id, vid_pid, manufacturer, product, serial_number, firmware_version
```

##### 5. 搜索可用端口 

```python
ports = sensor.search_ports()
# 返回: list[int] — 可用摄像头端口号列表（优先按 VID/PID 识别，失败时回退到 SN 匹配）
```

##### 6. 释放资源 

```python
sensor.release()
```

### SensorFunction — 数据后处理工具类

`SensorFunction` 是 `TouchSensor` 的后处理工具类，只处理 TouchSensor getter 已经返回的数据，不读取摄像头、不访问 Flash、不维护传感器内部状态。所有方法以"传入数据 → 返回分析结果"为主。

#### 初始化

```python
from api import SensorFunction

sf = SensorFunction(touch_thresholds=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `touch_thresholds` | `dict \| None` | `None` | 可选触摸阈值，支持 `contact`（默认 5.0）和 `slide`（默认 0.5） |

#### 主要方法

##### 1. 力方向分析 

```python
magnitude, angle_deg = sf.get_2DForce(fx_total, fy_total)
# 返回: (magnitude: float, angle_deg: float)
#       二维切向力模长与方向角（atan2(fy, fx)，单位度）

magnitude, azimuth_deg, pitch_deg = sf.get_3DForce(fx_total, fy_total, fz_total)
# 返回: (magnitude: float, azimuth_deg: float, pitch_deg: float)
#       三维力模长、x-y 平面方位角、相对 x-y 平面的俯仰角

cof = sf.get_COF(fx_total, fy_total, fz_total)
# 返回: float — 摩擦系数 = abs(sqrt(fx²+fy²) / fz)；fz=0 时返回 0
```

**使用场景：** 对 TouchSensor 输出的三轴力做几何分解，获取力的方向和大小。

##### 2. 触摸状态检测

```python
status, history = sf.detect_touch_status(
    fz_total, hue_matrix_filtered, thresholds=None, history=None, flow_matrix_filtered=None
)
# 返回: (status: str, history: dict)
#       status 为 "idle" | "contact" | "slide"
```

**判定规则：** `fz_total > contact` → 进入接触区；接触区内连续两帧 hue 光流合力 `f2d_hue > slide` 为 slide，否则 contact；`fz_total <= contact` 或 `hue_matrix_filtered` 为 None → idle。需调用方自行保存 `history` 字典以维持连续帧状态。

## 故障排除

### 问题1: 找不到 USB 摄像头

**症状**：USB 端口扫描对话框显示"无可用端口"，或 `search_ports()` 返回空列表。

**原因**：摄像头未正确连接、驱动未安装、或当前用户无设备访问权限。

**解决方案**：

1. 检查摄像头是否正确连接到 USB 端口
2. 在系统设备管理器中确认摄像头被识别
3. 尝试更换其他 USB 端口
4. Linux 下检查摄像头权限：

```bash
# Linux下检查摄像头设备是否存在
ls /dev/video*

# 添加当前用户到 video 用户组
sudo usermod -a -G video $USER
# 注销后重新登录使权限生效
```

5. Windows 下确认 `api/lib/windows_x86_64/SonixCamera.dll` 文件存在且未被杀毒软件拦截

---

### 问题2: ModuleNotFoundError

**症状**：`ModuleNotFoundError: No module named 'api'` 或 `No module named 'cv2'`。

**原因**：未从项目根目录运行，或依赖库未安装。

**解决方案**：

```bash
# 1. 确保从项目根目录运行
cd sightac_sdk
python GUI/demo_gui.py       # √ 正确

# 错误示例：
cd GUI
python demo_gui.py           # × 这样会找不到 api 模块

# 2. 如果缺少依赖，重新安装：
pip install -r requirements.txt
```

---

### 问题3: 摄像头初始化失败

**症状**：程序启动后报错 `Error: Can't fetch video stream` 或 `Can't open camera USB ID X`。

**原因**：摄像头被其他程序占用、usb_id 不正确、或 SDK 动态库缺失。

**解决方案**：

1. 确认摄像头没有被其他程序（如摄像头软件、浏览器）占用
2. 尝试不同的 `usb_id`（0, 1, 2...）
3. 更新OpenCV版本：pip install --upgrade opencv-python-headless
4. Windows下：安装DirectShow编解码器

---

### 问题4: 压力/力值显示异常

**症状**：压力热力图全红或无变化，三轴力值始终为 0 或剧烈跳动。

**原因**：基线校准不准、相机参数未固定、或环境光变化。

**解决方案**：

1. **重新校准**：确保传感器表面无压力（按 `校准` 键），或等待自动校准触发（连续约 60 帧低压力后自动执行）
2. **固定相机参数**：设置固定的曝光和白平衡值
3. **检查光照**：避免环境光频繁变化，使用稳定的照明条件
4. **检查 ROI**：确认 ROI 裁剪区域正确覆盖传感器有效区域

---

### 问题5: 光流检测不准确

**症状**：切向力场箭头方向杂乱、噪声过大，或力方向与实际不符。

**原因**：光流参数不适合当前场景、基线参考帧过时、或图像质量不稳定。

**解决方案**：

1. 重新校准传感器（按 `校准` 键）
2. 检查光照条件是否稳定，避免强光直射或阴影变化

---

### 问题6: 3D 可视化性能慢

**症状**：3D 表面更新卡顿，帧率明显低于 20 FPS。

**原因**：计算负载过高或 GPU 渲染性能不足。

**解决方案**：

确保系统支持 OpenGL 2.0+ 并有显卡驱动


## 📊 性能指标

| 指标 | 数值 |
|------|------|
| **帧率** | 25-30 FPS |
| **压力分辨率** | 取决于 ROI 裁剪区域（Flash 中配置） |
| **每帧处理延迟** | < 50ms |
| **内存占用** | ~240MB |
| **CPU占用** | 200 ~ 220%（curved多核） / 60 ~ 90%（bevel单核） |
| **GPU占用** | 2.0-4% |

**优化建议**：
- 使用GPU加速的OpenCV版本（`opencv-contrib-python`）
- 降低摄像头分辨率
- 减少网格密度

---

##  常见问题 (FAQ)

**Q1: 可以使用视频文件代替摄像头吗？**

A: 当前版本 `TouchSensor(usb_id)` 仅支持 USB 摄像头。如需使用视频文件，可参考底层源码自行扩展。

**Q2: 如何获取力数据矩阵？**

A: 调用 `update(mode=0)` 后使用 getter 方法：

```python
sensor.update(mode=0)
fx_mat, fy_mat, fz_mat = sensor.get_calib_matrix()  # 三轴逐像素力标定矩阵
fx, fy, fz = sensor.get_total_force()               # 三轴总力
```

**Q3: 可以同时连接多个传感器吗？**

A: 可以！为每个传感器分配不同的 `usb_id`，但是同一时刻只能运行一个端口：

```python
sensor1 = TouchSensor(usb_id=0)
sensor2 = TouchSensor(usb_id=1)
```

**Q4: 参数配置保存在哪里？**

A: 无需配置文件。力标定系数、ROI、设备类型等存储在摄像头 Flash 中，初始化时由 `read_flash()` 自动读取，按设备类型自动区分渲染模型。

**Q5: 如何提高检测灵敏度？**

A: 检测灵敏度由 SDK 内置参数按设备类型自动配置，如需调整，请联系开发人员调配参数。

---
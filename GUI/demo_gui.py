









import cv2
import os
import sys
import threading
from collections import deque
from threading import Lock

if sys.platform.startswith('linux'):
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from ctypes import c_void_p
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QFont, QPixmap, QPainter, QPen, QColor
from PyQt5 import QtCore
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api.algorithm.touch_sensor import TouchSensor

from radar import RadarPlotWidget, RadarComputeWorker
from usb_port import USBPortSelector, USBPortScanner
import time

GUI_VERSION = "2.4.1"

def _resource_path(rel):
    """定位随包资源文件。
    - PyInstaller 打包后：从临时解压目录 sys._MEIPASS 读取（对应 --add-data 的目标路径）；
    - 源码运行时：从仓库上级共享资源目录读取。
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, rel)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(repo_root), rel)

def _read_api_version():
    """直接 import 获取版本号。"""
    try:
        from api import VERSION
        return str(VERSION)
    except Exception:
        return "unknown"

API_VERSION = _read_api_version()


class LatestValueMailbox:
    """Keep the latest value without queuing one notification per publish.

    A successful notification claim remains pending until ``take``. ``None``
    denotes an empty mailbox and cannot be published.
    """

    def __init__(self):
        self._lock = Lock()
        self._value = None
        self._notification_pending = False
        self._closed = False

    def publish(self, value):
        """Replace the pending value; return whether the caller must notify."""
        if value is None:
            raise ValueError("None is reserved for an empty mailbox")
        with self._lock:
            if self._closed:
                return False
            self._value = value
            if self._notification_pending:
                return False
            self._notification_pending = True
            return True

    def take(self):
        """Take the latest value and permit the next publish to notify."""
        with self._lock:
            value = self._value
            self._value = None
            self._notification_pending = False
            return value

    def clear(self):
        """Discard the value while preserving any pending notification claim."""
        with self._lock:
            self._value = None

    def close(self):
        """Discard the value and permanently reject subsequent publishes."""
        with self._lock:
            self._closed = True
            self._value = None
            self._notification_pending = False


class FrameRateCounter:
    """Count completions in the trailing one and five seconds of a session."""

    def __init__(self, *, clock=time.perf_counter):
        self._clock = clock
        self._lock = threading.Lock()
        self._events = deque()
        self._started_at = clock()

    def _prune(self, now):
        while self._events and self._events[0] <= now - 5.0:
            self._events.popleft()

    def record(self):
        with self._lock:
            now = self._clock()
            self._prune(now)
            self._events.append(now)

    def reset(self):
        with self._lock:
            self._events.clear()
            self._started_at = self._clock()

    def snapshot(self):
        with self._lock:
            now = self._clock()
            self._prune(now)
            elapsed = max(now - self._started_at, 0.0)
            count = len(self._events)
            recent = sum(stamp > now - 1.0 for stamp in self._events)
            return {
                'fps': recent / min(elapsed, 1.0) if elapsed > 0 else 0.0,
                'avg_fps': count / min(elapsed, 5.0) if elapsed > 0 else 0.0,
                'warming_up': elapsed < 5.0,
                'count': count,
            }


class ThroughPoleOrbitMixin:
    """允许相机从模型顶部翻转到模型底部"""

    CAMERA_ELEVATION_LIMIT = 270.0

    def orbit(self, azim, elev):
        if self.opts['rotationMethod'] == 'quaternion':
            q = QtGui.QQuaternion.fromEulerAngles(elev, -azim, 0)
            q *= self.opts['rotation']
            self.opts['rotation'] = q
        else:
            self.opts['azimuth'] += azim
            elevation = self.opts['elevation'] + elev
            self.opts['elevation'] = max(
                -self.CAMERA_ELEVATION_LIMIT,
                min(self.CAMERA_ELEVATION_LIMIT, elevation)
            )

        self.update()

def _resolve_device_profile(device_type):
    """Return the display labels for a supported device type."""
    if device_type == 'curved':
        return '曲面', '指尖'
    if device_type == 'bevel':
        return '斜面', '平面'
    raise ValueError(
        f"未识别的设备类型 (device_type={device_type!r})。\n\n"
        "请检查设备是否已烧录正确的设备标识。"
    )

class SolidSurfaceItem(GLGraphicsItem):
    def __init__(self, color=(0.0, 0.0, 0.0, 1.0)):
        super().__init__()
        self.setGLOptions('opaque')
        self.color = np.ascontiguousarray(color, dtype=np.float32)
        self.vertices = None
        self.indices = None
        self._geometry_dirty = True
        self._initialized = False
        self._program = None
        self._vbo_vertices = None
        self._ebo = None
        self._index_count = 0

    def set_geometry(self, vertices, faces):
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        indices = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
        if (
                self.vertices is not None and
                self.vertices.shape == vertices.shape and
                self.indices.shape == indices.shape and
                np.array_equal(self.vertices, vertices) and
                np.array_equal(self.indices, indices)
        ):
            return
        self.vertices = vertices
        self.indices = indices
        self._index_count = int(indices.size)
        self._geometry_dirty = True
        self.update()

    def _ensure_initialized(self):
        if self._initialized:
            return
        vertex_shader = """
            #version 120
            attribute vec3 a_position;
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * vec4(a_position, 1.0);
            }
        """
        fragment_shader = """
            #version 120
            uniform vec4 u_color;
            void main() {
                gl_FragColor = u_color;
            }
        """
        self._program = compileProgram(
            compileShader(vertex_shader, GL.GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL.GL_FRAGMENT_SHADER)
        )
        self._vbo_vertices = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)
        self._initialized = True

    def paint(self):
        if self.vertices is None or self._index_count <= 0:
            return
        self.setupGLState()
        self._ensure_initialized()

        if self._geometry_dirty:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_vertices)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL.GL_STATIC_DRAW)
            self._geometry_dirty = False

        GL.glUseProgram(self._program)
        GL.glUniform4f(
            GL.glGetUniformLocation(self._program, "u_color"),
            float(self.color[0]), float(self.color[1]), float(self.color[2]), float(self.color[3])
        )

        pos_loc = GL.glGetAttribLocation(self._program, "a_position")
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_vertices)
        GL.glEnableVertexAttribArray(pos_loc)
        GL.glVertexAttribPointer(pos_loc, 3, GL.GL_FLOAT, False, 0, None)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDrawElements(GL.GL_TRIANGLES, self._index_count, GL.GL_UNSIGNED_INT, None)

        GL.glDisableVertexAttribArray(pos_loc)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)

class InstancedConeItem(GLGraphicsItem):
    def __init__(self, cols=12):
        super().__init__()
        self.setGLOptions('translucent')
        self.cols = int(cols)
        self.instances = None
        self._instance_count = 0
        self._instances_dirty = True
        self._initialized = False
        self._program = None
        self._vbo_base = None
        self._vbo_instances = None
        self._ebo = None
        self._base_vertices, self._indices = self._build_cone_template()

    def _build_cone_template(self):
        verts = [[0.0, 0.0, 0.0]]
        for k in range(self.cols):
            angle = 2.0 * np.pi * k / self.cols
            verts.append([float(np.cos(angle)), float(np.sin(angle)), -1.0])
        verts.append([0.0, 0.0, -1.0])
        center_idx = self.cols + 1
        faces = []
        for k in range(self.cols):
            a = 1 + k
            b = 1 + ((k + 1) % self.cols)
            faces.extend([[0, a, b], [center_idx, b, a]])
        return (
            np.ascontiguousarray(verts, dtype=np.float32),
            np.ascontiguousarray(np.asarray(faces, dtype=np.uint32).reshape(-1), dtype=np.uint32)
        )

    def set_instances(self, positions, directions, scales, colors):
        if len(positions) == 0:
            self.instances = None
            self._instance_count = 0
            self._instances_dirty = True
            self.hide()
            return

        positions = np.ascontiguousarray(positions, dtype=np.float32)
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        scales = np.ascontiguousarray(scales.reshape(-1, 1), dtype=np.float32)
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        default_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        fallback_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        refs = np.repeat(default_ref[None, :], len(directions), axis=0)
        near_parallel = np.abs(directions @ default_ref) >= 0.92
        refs[near_parallel] = fallback_ref

        u = np.cross(directions, refs)
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
        v = np.cross(u, directions)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)

        self.instances = np.ascontiguousarray(
            np.column_stack([positions, directions, u, v, scales, colors]),
            dtype=np.float32
        )
        self._instance_count = int(len(positions))
        self._instances_dirty = True
        self.show()
        self.update()

    def _ensure_initialized(self):
        if self._initialized:
            return
        vertex_shader = """
            #version 120
            attribute vec3 a_local;
            attribute vec3 a_pos;
            attribute vec3 a_dir;
            attribute vec3 a_u;
            attribute vec3 a_v;
            attribute float a_scale;
            attribute vec4 a_color;
            varying vec4 v_color;

            void main() {
                float radius = 3.0 * a_scale;
                float length = 5.4 * a_scale;
                vec3 world = a_pos
                    + a_local.x * radius * a_u
                    + a_local.y * radius * a_v
                    + a_local.z * length * a_dir;
                gl_Position = gl_ModelViewProjectionMatrix * vec4(world, 1.0);
                v_color = a_color;
            }
        """
        fragment_shader = """
            #version 120
            varying vec4 v_color;
            void main() {
                gl_FragColor = v_color;
            }
        """
        self._program = compileProgram(
            compileShader(vertex_shader, GL.GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL.GL_FRAGMENT_SHADER)
        )
        self._vbo_base = GL.glGenBuffers(1)
        self._vbo_instances = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_base)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._base_vertices.nbytes, self._base_vertices, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self._indices.nbytes, self._indices, GL.GL_STATIC_DRAW)
        self._initialized = True

    def _set_instance_attrib(self, name, size, offset_floats, stride_bytes):
        loc = GL.glGetAttribLocation(self._program, name)
        if loc < 0:
            return loc
        GL.glEnableVertexAttribArray(loc)
        GL.glVertexAttribPointer(
            loc, size, GL.GL_FLOAT, False,
            stride_bytes, c_void_p(offset_floats * 4)
        )
        GL.glVertexAttribDivisor(loc, 1)
        return loc

    def paint(self):
        if self.instances is None or self._instance_count <= 0:
            return
        self.setupGLState()
        self._ensure_initialized()

        GL.glUseProgram(self._program)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_base)
        local_loc = GL.glGetAttribLocation(self._program, "a_local")
        GL.glEnableVertexAttribArray(local_loc)
        GL.glVertexAttribPointer(local_loc, 3, GL.GL_FLOAT, False, 0, None)
        GL.glVertexAttribDivisor(local_loc, 0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_instances)
        if self._instances_dirty:
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.instances.nbytes, self.instances, GL.GL_STREAM_DRAW)
            self._instances_dirty = False

        stride = self.instances.shape[1] * 4
        locs = [
            self._set_instance_attrib("a_pos", 3, 0, stride),
            self._set_instance_attrib("a_dir", 3, 3, stride),
            self._set_instance_attrib("a_u", 3, 6, stride),
            self._set_instance_attrib("a_v", 3, 9, stride),
            self._set_instance_attrib("a_scale", 1, 12, stride),
            self._set_instance_attrib("a_color", 4, 13, stride),
        ]

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDrawElementsInstanced(
            GL.GL_TRIANGLES, int(self._indices.size),
            GL.GL_UNSIGNED_INT, None, self._instance_count
        )

        GL.glDisableVertexAttribArray(local_loc)
        for loc in locs:
            if loc >= 0:
                GL.glVertexAttribDivisor(loc, 0)
                GL.glDisableVertexAttribArray(loc)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)

class SensorUpdateWorker(QtCore.QThread):
    """Update the sensor and notify the GUI of the latest completed snapshot."""
    frame_available = QtCore.pyqtSignal()
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, sensor):
        super().__init__()
        self.sensor = sensor
        self._frame_mailbox = LatestValueMailbox()
        self.seq = 0
        
        
        self._last_frame_data = None   
        self._consecutive_same_frames = 0

    def take_latest_frame(self):
        return self._frame_mailbox.take()

    def close_mailbox(self):
        self._frame_mailbox.close()

    @staticmethod
    def _copy_matrix(matrix):
        if matrix is None:
            return None
        return np.array(matrix, copy=True)

    def run(self):
        while not self.isInterruptionRequested():
            
            try:
                self.sensor.update(mode=0)
                
                current_frame = self.sensor.preprocessed_frame
                if current_frame is not None:
                    if self._last_frame_data is not None and np.array_equal(current_frame, self._last_frame_data):
                        self.error_occurred.emit("相机连接已断开，程序即将退出")
                        break
                    self._last_frame_data = current_frame.copy()
                
                fx_t, fy_t, fz_t = self.sensor.get_total_force()
                fx_matrix = self._copy_matrix(self.sensor.fx_matrix)
                fy_matrix = self._copy_matrix(self.sensor.fy_matrix)
                _, _, fz_matrix = self.sensor.get_calib_matrix()
                fz_matrix = self._copy_matrix(fz_matrix)
                hue_matrix = self._copy_matrix(self.sensor.get_hue_matrix())
                flow_matrix = self._copy_matrix(self.sensor.get_flow_matrix())

                if fz_matrix is None or fx_matrix is None or fy_matrix is None:
                    continue

                self.seq += 1
                
                now = time.perf_counter()
                
                
                
                
                

                packet = {
                    'seq': self.seq,
                    'fx_t': 0 if fx_t is None else int(fx_t),
                    'fy_t': 0 if fy_t is None else int(fy_t),
                    'fz_t': 0 if fz_t is None else int(fz_t),
                    'fx_raw': self.sensor.fx_total,
                    'fy_raw': self.sensor.fy_total,
                    'fz_raw': self.sensor.fz_total,
                    'raw_frame': self._copy_matrix(self.sensor.preprocessed_frame),
                    'fx_matrix': fx_matrix,
                    'fy_matrix': fy_matrix,
                    'fz_matrix': fz_matrix,
                    'hue_matrix': hue_matrix,
                    'flow_matrix': flow_matrix,
                    'contact_threshold': float(self.sensor.auto_calibration_fz_threshold),
                    
                    
                    'timestamp': now,
                    'autocal_count': self.sensor.auto_calibration_frame_count,
                    'autocal_spike': self.sensor.auto_calibration_spike_count,
                }
                if self._frame_mailbox.publish(packet):
                    self.frame_available.emit()
            except Exception as e:
                self.error_occurred.emit(str(e))
                break


class RadarFeatureWorker(QtCore.QThread):
    """雷达特征计算线程：只处理最新输入帧（忙时丢帧，latest-wins），
    结果通过合并通知交给主线程，避免 GUI 线程承担 numpy 计算。"""

    result_available = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.radar_compute = RadarComputeWorker()
        self._result_mailbox = LatestValueMailbox()
        self._input = None
        self._input_lock = threading.Lock()
        self._input_ready = threading.Event()
        self._reset_requested = False
        self._input_closed = False

    def take_latest_result(self):
        return self._result_mailbox.take()

    def close_mailbox(self):
        self._result_mailbox.close()
        with self._input_lock:
            self._input_closed = True
            self._input = None
            self._reset_requested = False

    def submit(self, frame_data):
        """主线程提交最新帧；worker 忙时旧输入被覆盖（latest-wins）。"""
        with self._input_lock:
            if self._input_closed:
                return
            self._input = frame_data
            self._input_ready.set()

    def request_reset(self):
        """请求在 worker 线程内清空 RadarComputeWorker 跨帧状态（校准/换设备后）。"""
        self._result_mailbox.clear()
        with self._input_lock:
            if self._input_closed:
                return
            self._reset_requested = True
            self._input = None
            self._input_ready.set()

    def wake(self):
        self._input_ready.set()

    def run(self):
        while not self.isInterruptionRequested():
            self._input_ready.wait(0.05)
            with self._input_lock:
                reset_requested = self._reset_requested
                self._reset_requested = False
                frame = self._input
                self._input = None
                self._input_ready.clear()
            if reset_requested:
                self.radar_compute.reset()
            if frame is None:
                continue
            try:
                t0 = time.perf_counter()
                xs, ys, features = self.radar_compute.compute(
                    frame["fz"],
                    frame["hue_matrix"],
                    frame["flow_matrix"],
                    frame["fx_matrix"],
                    frame["fy_matrix"],
                    frame["fz_matrix"],
                    frame["timestamp"],
                    frame["contact_threshold"],
                )
                compute_ms = (time.perf_counter() - t0) * 1000.0
                result = {
                    "generation": frame["generation"],
                    "frame_id": frame["frame_id"],
                    "xs": xs,
                    "ys": ys,
                    "features": features,
                    "compute_ms": compute_ms,
                }
                if self._result_mailbox.publish(result):
                    self.result_available.emit()
            except Exception as exc:
                print(f"[RadarWorker] 计算错误: {exc}")

class StartupPortScanWorker(QtCore.QThread):
    """Background worker that scans available USB ports for startup auto-connect."""

    scan_finished = QtCore.pyqtSignal(int, object, str)

    def __init__(self, request_id):
        super().__init__()
        self.request_id = request_id

    def run(self):
        scanner = USBPortScanner()
        try:
            ports = scanner.scan_available_ports()
            available_port_numbers = sorted(
                int(port_info.port)
                for port_info in ports
                if getattr(port_info, 'is_available', False)
            )
            self.scan_finished.emit(self.request_id, available_port_numbers, "")
        except Exception as e:
            self.scan_finished.emit(self.request_id, [], str(e))
        finally:
            scanner.release_all_ports()


class SensorInitWorker(QtCore.QThread):
    """后台线程构造 TouchSensor（camera_init + calibrate_baseline），
    避免初启动时主线程被阻塞导致界面冻结。

    注意：连接期间“更换设备”按钮保持可用。用户在构造过程中发起更换时，
    结果经 request_id 判定为 stale，由主线程槽释放孤儿传感器后自动续跑
    更换流程（见 _resume_pending_manual_connect）。"""

    initialized = QtCore.pyqtSignal(int, object)   
    failed = QtCore.pyqtSignal(int, str)           

    def __init__(self, usb_id, request_id):
        super().__init__()
        self.usb_id = usb_id
        self.request_id = request_id

    def run(self):
        sensor_obj = None
        try:
            sensor_obj = TouchSensor(usb_id=self.usb_id)
        except Exception as e:
            if sensor_obj is not None:
                try:
                    sensor_obj.release()
                except Exception:
                    pass
            self.failed.emit(self.request_id, str(e))
            return
        self.initialized.emit(self.request_id, sensor_obj)


class FingertipBoardGLWidget(ThroughPoleOrbitMixin, gl.GLViewWidget):
    """指尖模式 OpenGL 视图：在 XY 面和 XZ 面的 1/4 球面上绘制力箭头"""

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((0, 0, 0, 255))
        self.reset_camera_view()
        self.board_w = 0
        self.board_h = 0

        
        self.fingertip_radius = 100.0  
        self.fingertip_stretch = 1.8  
        self.fingertip_stretch_y = 1.5  
        self.fingertip_segments_u = 30  
        self.fingertip_segments_v = 30  

        

        
        self.fingertip_arrow_xy_scale = 60000.0  
        self.fingertip_arrow_z_scale = 6000.0  
        self.fingertip_arrow_color_gain = 500  
        self.fingertip_arrow_min_force = 0.03  
        self.fingertip_arrow_release_force = 0.018  
        self.fingertip_arrow_smoothing_alpha = 0.80  
        self.fingertip_arrow_length_gain = 15.0  
        self.fingertip_arrow_min_len = 20.0  
        self.fingertip_arrow_width_scale = 0.5  
        self.fingertip_arrow_line_width = 4.0  
        self.fingertip_arrow_head_base = 0.5  
        self.fingertip_arrow_head_scale = 0.1  
        self.fingertip_arrow_head_max = 0.5  
        self.fingertip_arrow_surface_offset = 3.0  
        self.fingertip_point_size = 5.0
        self.fingertip_point_offset = 1.2
        self.fingertip_point_idle_color = np.array((0.70, 0.74, 0.82, 0.95), dtype=np.float32)

        
        dummy = np.zeros((2, 2, 4), dtype=np.uint8)
        self.image_item = gl.GLImageItem(data=dummy)
        self.addItem(self.image_item)
        self.surface_mesh_item = SolidSurfaceItem(color=(0.0, 0.0, 0.0, 1.0))
        self.surface_mesh_item.hide()
        self.addItem(self.surface_mesh_item)
        self._surface_body_cache_key = None
        self._surface_body_cache = None
        self._surface_body_uploaded_key = None
        self.surface_points_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            size=self.fingertip_point_size,
            pxMode=True
        )
        self.surface_points_item.setGLOptions('translucent')
        self.surface_points_item.hide()
        self.addItem(self.surface_points_item)

        
        arrow_overlay_gl_options = {
            GL.GL_DEPTH_TEST: False,
            GL.GL_BLEND: True,
            GL.GL_ALPHA_TEST: False,
            GL.GL_CULL_FACE: False,
            'glBlendFunc': (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
        }
        self.batched_arrow_lines = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            width=2.0,
            mode='lines',
            antialias=False
        )
        self.batched_arrow_lines.setGLOptions(arrow_overlay_gl_options)
        self.addItem(self.batched_arrow_lines)
        self.arrow_head_cols = 12
        self.batched_arrow_heads_item = InstancedConeItem(cols=self.arrow_head_cols)
        self.batched_arrow_heads_item.setGLOptions(arrow_overlay_gl_options)
        self.batched_arrow_heads_item.hide()
        self.addItem(self.batched_arrow_heads_item)
        self.arrow_heads = []
        self.arrow_head_mesh = gl.MeshData.cylinder(rows=2, cols=12, radius=[3, 0.0], length=5.4)
        self._max_arrows = 256  
        self._surface_cache_key = None
        self._surface_cache = None
        self._fingertip_force_smooth = None
        self._fingertip_active_mask = None
        self._fingertip_visual_params_key = None

        jet_lut_bgr = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(256, 3)
        self.jet_lut_rgba = np.zeros((256, 4), dtype=np.float32)
        self.jet_lut_rgba[:, 0] = jet_lut_bgr[:, 2] / 255.0
        self.jet_lut_rgba[:, 1] = jet_lut_bgr[:, 1] / 255.0
        self.jet_lut_rgba[:, 2] = jet_lut_bgr[:, 0] / 255.0
        self.jet_lut_rgba[:, 3] = 1.0

    def reset_camera_view(self):
        self.opts['fov'] = 60
        self.setCameraPosition(
            pos=QtGui.QVector3D(0, 50, 0),
            distance=400,
            elevation=90,
            azimuth=-90
        )

    def _get_surface_cache(self, w, h, grid_nx=16, grid_ny=16):
        key = (
            int(w), int(h), int(grid_nx), int(grid_ny),
            float(self.fingertip_radius),
            float(self.fingertip_stretch),
            float(self.fingertip_stretch_y),
        )
        if self._surface_cache_key == key and self._surface_cache is not None:
            return self._surface_cache

        gx_1d = np.linspace(0, w - 1, grid_nx).astype(np.int32)
        gy_1d = np.linspace(0, h - 1, grid_ny).astype(np.int32)
        gx, gy = np.meshgrid(gx_1d, gy_1d)
        cx = gx.reshape(-1).astype(np.int32)
        cy = gy.reshape(-1).astype(np.int32)

        u = cx.astype(np.float32) / max(w - 1, 1)
        v = cy.astype(np.float32) / max(h - 1, 1)
        beta = u * np.pi
        alpha_start = (v ** 1.8) * (np.pi / 2)
        alpha_basis = v * (np.pi / 2)

        R = self.fingertip_radius
        stretch = self.fingertip_stretch
        stretch_y = self.fingertip_stretch_y

        cos_start = np.cos(alpha_start)
        start_pts = np.column_stack([
            -R * cos_start * np.cos(beta),
            R * stretch_y * np.sin(alpha_start),
            R * cos_start * np.sin(beta),
        ]).astype(np.float32)

        cos_a = np.cos(alpha_basis)
        sin_a = np.sin(alpha_basis)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)

        normals = np.column_stack([
            -cos_a * cos_b,
            sin_a / (stretch_y * stretch_y),
            cos_a * sin_b / (stretch * stretch),
        ]).astype(np.float32)
        normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(normal_norms, 1e-6)

        d_alpha = np.column_stack([
            R * sin_a * cos_b,
            R * stretch_y * cos_a,
            -R * sin_a * sin_b * stretch,
        ]).astype(np.float32)
        d_alpha /= np.maximum(np.linalg.norm(d_alpha, axis=1, keepdims=True), 1e-6)

        d_beta = np.column_stack([
            R * cos_a * sin_b,
            np.zeros_like(cos_a),
            R * cos_a * cos_b * stretch,
        ]).astype(np.float32)
        d_beta /= np.maximum(np.linalg.norm(d_beta, axis=1, keepdims=True), 1e-6)

        half_win = max(1, int(round((w / grid_nx) / 2)))
        y0 = np.maximum(0, cy - half_win)
        y1 = np.minimum(h, cy + half_win + 1)
        x0 = np.maximum(0, cx - half_win)
        x1 = np.minimum(w, cx + half_win + 1)
        area = np.maximum((y1 - y0) * (x1 - x0), 1).astype(np.float32)

        self._surface_cache_key = key
        self._surface_cache = {
            'start_pts': start_pts,
            'normals': normals,
            'd_alpha': d_alpha,
            'd_beta': d_beta,
            'x0': x0,
            'x1': x1,
            'y0': y0,
            'y1': y1,
            'area': area,
            'count': len(cx),
        }
        return self._surface_cache

    @staticmethod
    def _window_means(matrix, cache):
        integral = np.pad(
            matrix.astype(np.float32, copy=False).cumsum(axis=0).cumsum(axis=1),
            ((1, 0), (1, 0)),
            mode='constant'
        )
        sums = (
            integral[cache['y1'], cache['x1']]
            - integral[cache['y0'], cache['x1']]
            - integral[cache['y1'], cache['x0']]
            + integral[cache['y0'], cache['x0']]
        )
        return (sums / cache['area']).astype(np.float32)

    def _get_surface_body_cache(self, w, h):
        key = (
            int(w), int(h),
            int(self.fingertip_segments_u),
            int(self.fingertip_segments_v),
            float(self.fingertip_radius),
            float(self.fingertip_stretch_y),
        )
        if self._surface_body_cache_key == key and self._surface_body_cache is not None:
            return self._surface_body_cache

        u_seg = self.fingertip_segments_u
        v_seg = self.fingertip_segments_v
        R = self.fingertip_radius
        stretch_y = self.fingertip_stretch_y

        verts = []
        for j in range(v_seg + 1):
            v = j / max(v_seg, 1)
            alpha = (v ** 1.8) * (np.pi / 2)
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            for i in range(u_seg + 1):
                u = i / max(u_seg, 1)
                beta = u * np.pi
                verts.append([
                    -R * cos_a * np.cos(beta),
                    R * stretch_y * sin_a,
                    R * cos_a * np.sin(beta),
                ])

        faces = []
        row = u_seg + 1
        for j in range(v_seg):
            for i in range(u_seg):
                a = j * row + i
                b = a + 1
                c = (j + 1) * row + i
                d = c + 1
                faces.extend([[a, b, d], [a, d, c]])

        self._surface_body_cache_key = key
        self._surface_body_cache = {
            'verts': np.asarray(verts, dtype=np.float32),
            'faces': np.asarray(faces, dtype=np.uint32),
        }
        return self._surface_body_cache

    def _update_surface_body(self, fz_calib):
        cache = self._get_surface_body_cache(fz_calib.shape[1], fz_calib.shape[0])
        if self._surface_body_uploaded_key != self._surface_body_cache_key:
            self.surface_mesh_item.set_geometry(cache['verts'], cache['faces'])
            self._surface_body_uploaded_key = self._surface_body_cache_key
        self.surface_mesh_item.show()

    def _clear_arrow_items(self):
        self.batched_arrow_lines.setData(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            width=2.0,
            mode='lines'
        )
    def _clear_arrow_heads(self):
        """Hide all fingertip arrow heads."""
        self.batched_arrow_heads_item.hide()
        for cone in self.arrow_heads:
            cone.hide()

    def _set_batched_arrow_heads(self, arrow_data_list):
        positions = np.asarray([data['pos'] for data in arrow_data_list], dtype=np.float32)
        directions = np.asarray([data['direction'] for data in arrow_data_list], dtype=np.float32)
        scales = np.asarray([data['scale'] for data in arrow_data_list], dtype=np.float32)
        colors = np.asarray([
            data.get('color', (1.0, 1.0, 0.0, 0.95)) for data in arrow_data_list
        ], dtype=np.float32)

        dir_norms = np.linalg.norm(directions, axis=1)
        valid = dir_norms > 1e-6
        if not np.any(valid):
            self.batched_arrow_heads_item.hide()
            return

        positions = positions[valid]
        directions = directions[valid] / dir_norms[valid, None]
        scales = scales[valid]
        colors = colors[valid]
        self.batched_arrow_heads_item.set_instances(positions, directions, scales, colors)

    def _update_arrow_heads(self, arrow_data_list):
        """更新3D箭头头部（圆锥体，复用对象池）"""
        if not arrow_data_list:
            self._clear_arrow_heads()
            return

        self._set_batched_arrow_heads(arrow_data_list)

    def set_planar_arrow_visual_params(
            self,
            arrow_xy_scale,
            arrow_z_scale,
            arrow_color_gain,
            arrow_min_force,
            arrow_min_len,
            xy_world_scale=1.0,
            z_world_scale=None
    ):
        """让指尖模式沿用平面模式的箭头可视化系数，并做球面单位换算。"""
        if z_world_scale is None:
            z_world_scale = xy_world_scale
        xy_unit_scale = 100.0 * float(xy_world_scale)  
        z_unit_scale = 100.0 * float(z_world_scale)
        self.fingertip_arrow_xy_scale = float(arrow_xy_scale) * xy_unit_scale
        self.fingertip_arrow_z_scale = float(arrow_z_scale) * z_unit_scale
        self.fingertip_arrow_color_gain = float(arrow_color_gain)
        self.fingertip_arrow_min_force = float(arrow_min_force)
        self.fingertip_arrow_release_force = float(arrow_min_force) * 0.6
        self.fingertip_arrow_min_len = float(arrow_min_len) * z_unit_scale
        self.fingertip_arrow_line_width = 4.0

    def set_fingertip_data(self, fz_calib, fx_calib, fy_calib):
        """
        在指尖球面上绘制力箭头。
        使用指尖模式专用模长参数（self.fingertip_arrow_*）。
        - fz_calib: 法向压力矩阵
        - fx_calib: X方向切向力矩阵
        - fy_calib: Y方向切向力矩阵
        """
        if fz_calib is None or fx_calib is None or fy_calib is None:
            return

        
        fz_calib = cv2.flip(fz_calib, 0)
        fx_calib = cv2.flip(fx_calib, 0)
        fy_calib = cv2.flip(fy_calib, 0)

        
        arrow_xy_scale = self.fingertip_arrow_xy_scale
        arrow_z_scale = self.fingertip_arrow_z_scale
        arrow_color_gain = self.fingertip_arrow_color_gain
        arrow_min_force = self.fingertip_arrow_min_force
        arrow_min_len = self.fingertip_arrow_min_len
        arrow_head_base = self.fingertip_arrow_head_base
        arrow_head_scale = self.fingertip_arrow_head_scale
        arrow_head_max = self.fingertip_arrow_head_max

        visual_params_key = (
            float(arrow_xy_scale),
            float(arrow_z_scale),
            float(arrow_color_gain),
            float(arrow_min_force),
            float(self.fingertip_arrow_release_force),
            float(self.fingertip_arrow_smoothing_alpha),
            float(self.fingertip_arrow_length_gain),
            float(arrow_min_len),
            float(self.fingertip_arrow_line_width),
            float(arrow_head_base),
            float(arrow_head_scale),
            float(arrow_head_max),
            float(self.fingertip_arrow_surface_offset),
        )
        if self._fingertip_visual_params_key != visual_params_key:
            self._fingertip_force_smooth = None
            self._fingertip_active_mask = None
            self._fingertip_visual_params_key = visual_params_key

        h, w = fz_calib.shape[:2]
        self._update_surface_body(fz_calib)
        grid_nx, grid_ny = 16, 16
        cache = self._get_surface_cache(w, h, grid_nx, grid_ny)
        fx_vals = self._window_means(fx_calib, cache)
        fy_vals = self._window_means(fy_calib, cache)
        fz_vals = self._window_means(fz_calib, cache)

        force_vals = np.column_stack([fx_vals, fy_vals, fz_vals]).astype(np.float32)
        if self._fingertip_force_smooth is None or self._fingertip_force_smooth.shape != force_vals.shape:
            smooth_force_vals = force_vals
        else:
            alpha = float(np.clip(self.fingertip_arrow_smoothing_alpha, 0.0, 1.0))
            smooth_force_vals = alpha * force_vals + (1.0 - alpha) * self._fingertip_force_smooth
        self._fingertip_force_smooth = smooth_force_vals
        fx_vals = smooth_force_vals[:, 0]
        fy_vals = smooth_force_vals[:, 1]
        fz_vals = smooth_force_vals[:, 2]

        mag_xy = np.hypot(fx_vals, fy_vals)
        mag_3d = np.hypot(mag_xy, fz_vals)
        start_pts = cache['start_pts']
        normals = cache['normals']
        d_alpha = cache['d_alpha']
        d_beta = cache['d_beta']

        end_pts = start_pts + normals * (arrow_min_len * 0.01)
        if self._fingertip_active_mask is None or self._fingertip_active_mask.shape != mag_3d.shape:
            active = mag_3d >= arrow_min_force
        else:
            release_force = min(arrow_min_force, float(self.fingertip_arrow_release_force))
            active = np.where(
                self._fingertip_active_mask,
                mag_3d >= release_force,
                mag_3d >= arrow_min_force
            )
        self._fingertip_active_mask = active
        if np.any(active):
            end_pts = end_pts.copy()
            tangent_vec = (
                d_beta[active] * (fx_vals[active] * arrow_xy_scale * 0.01)[:, None]
                + d_alpha[active] * (fy_vals[active] * arrow_xy_scale * 0.01)[:, None]
            )
            normal_vec = normals[active] * (fz_vals[active] * arrow_z_scale * 0.01)[:, None]
            raw_vec = tangent_vec + normal_vec
            raw_len = np.linalg.norm(raw_vec, axis=1)

            
            
            active_mag = mag_3d[active]
            excess_ratio = np.clip(
                (active_mag - arrow_min_force) / np.maximum(active_mag, 1e-6),
                0.0,
                1.0
            )
            min_len_world = arrow_min_len * 0.01
            display_len = (
                min_len_world
                + raw_len * excess_ratio * float(self.fingertip_arrow_length_gain)
            )

            
            
            display_vec = normals[active] * min_len_world
            valid = raw_len > 1e-6
            if np.any(valid):
                display_vec[valid] = (
                    raw_vec[valid]
                    * (display_len[valid] / raw_len[valid])[:, None]
                )
            end_pts[active] = start_pts[active] + display_vec
            

        color_levels = np.clip(mag_3d * arrow_color_gain, 0, 230).astype(np.int32)
        colors = self.jet_lut_rgba[color_levels].copy()
        colors[:, :3] = np.maximum(colors[:, :3], 0.35)
        colors[:, 3] = 1.0

        num_arrows = cache['count']
        point_pts = start_pts + normals * self.fingertip_point_offset
        point_colors = np.repeat(self.fingertip_point_idle_color[None, :], num_arrows, axis=0)
        if np.any(active):
            point_colors[active] = colors[active]
        self.surface_points_item.setData(
            pos=point_pts,
            color=np.ascontiguousarray(point_colors, dtype=np.float32),
            size=self.fingertip_point_size,
            pxMode=True
        )
        self.surface_points_item.show()

        
        if np.any(active):
            arrow_offset = normals * self.fingertip_arrow_surface_offset
            act_start = start_pts[active] + arrow_offset[active]
            act_end = end_pts[active] + arrow_offset[active]
            act_colors = colors[active]
            n_active = int(np.sum(active))

            vecs = act_end - act_start
            lengths = np.linalg.norm(vecs, axis=1)

            line_pts = np.empty((n_active * 2, 3), dtype=np.float32)
            line_colors_arr = np.empty((n_active * 2, 4), dtype=np.float32)
            line_pts[0::2] = act_start
            line_pts[1::2] = act_end
            line_colors_arr[0::2] = act_colors
            line_colors_arr[1::2] = act_colors
            self.batched_arrow_lines.setData(
                pos=line_pts,
                color=line_colors_arr,
                width=float(self.fingertip_arrow_line_width),
                mode='lines'
            )

            arrow_heads_data = []
            for end_pt, vec, fn, rgba in zip(act_end, vecs, lengths, act_colors):
                if fn > 0.01:
                    head_scale = arrow_head_base + min(float(fn) * arrow_head_scale, arrow_head_max)
                    arrow_heads_data.append({
                        'pos': end_pt,
                        'direction': vec / float(fn),
                        'scale': head_scale,
                        'color': tuple(float(c) for c in rgba)
                    })

            self._update_arrow_heads(arrow_heads_data)
        else:
            self._clear_arrow_items()
            self._clear_arrow_heads()

        return

    def reset_view_data(self):
        self._fingertip_force_smooth = None
        self._fingertip_active_mask = None
        self._clear_arrow_items()
        self._clear_arrow_heads()
        self.surface_mesh_item.hide()
        self.surface_points_item.hide()
        self.surface_points_item.setData(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            size=self.fingertip_point_size,
            pxMode=True
        )
        dummy = np.zeros((2, 2, 4), dtype=np.uint8)
        self.image_item.setData(data=dummy)

class PlanarAxesOverlay(QtWidgets.QWidget):
    """独立绘制平面视图坐标轴，避免在 QOpenGLWidget 的 paintEvent 内混用 QPainter。"""

    def __init__(self, gl_view):
        super().__init__(gl_view)
        self.gl_view = gl_view
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        try:
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.fillRect(event.rect(), QtCore.Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            self.gl_view._draw_overlay_axes(painter)
        finally:
            painter.end()


class PlanarBoardGLWidget(ThroughPoleOrbitMixin, gl.GLViewWidget):
    """OpenGL 平板视图（适配新版 touch_sensor API）"""

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((0, 0, 0, 255))
        self.reset_camera_view()
        self.board_w = 0
        self.board_h = 0
        self.board_thickness = 24.0

        self.profile_base_half = 0.5
        self.profile_corner_r = 0.16
        self.profile_fillet_z = 0.22
        self.profile_corner_seg = 10
        self.profile_z_steps = 18
        self.display_scale = 1.22
        self.overlay_to_board_ratio = 0.95
        self.overlay_z_offset = 0.06
        self.arrow_z_base = 0.12

        
        self._cached_tw = 0
        self._cached_th = 0
        self._cached_corner_px = 0.0
        self._rounded_mask_cache = {}

        cube_md = self._build_cube_meshdata()
        self.board_body = gl.GLMeshItem(
            meshdata=cube_md,
            color=(0.55, 0.58, 0.63, 1.0),
            smooth=False,
            drawEdges=False
        )
        self.addItem(self.board_body)

        dummy = np.zeros((2, 2, 4), dtype=np.uint8)
        self.image_item = gl.GLImageItem(data=dummy)
        self.addItem(self.image_item)

        
        self.batched_arrow_lines = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=np.empty((0, 4), dtype=np.float32),
            width=2.0,
            mode='lines',
            antialias=False
        )
        self.batched_arrow_lines.setGLOptions('opaque')
        self.addItem(self.batched_arrow_lines)

        
        self.batched_arrow_heads_item = InstancedConeItem(cols=12)
        self.batched_arrow_heads_item.hide()
        self.addItem(self.batched_arrow_heads_item)

        
        self.arrow_items = []  
        self.arrow_heads = []  
        self.arrow_head_mesh = gl.MeshData.cylinder(rows=2, cols=12, radius=[3, 0.0], length=5.4)

        
        self.arrow_width_scale = 0.1

        jet_lut_bgr = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(256, 3)
        self.jet_lut_rgba = np.zeros((256, 4), dtype=np.float32)
        self.jet_lut_rgba[:, 0] = jet_lut_bgr[:, 2] / 255.0
        self.jet_lut_rgba[:, 1] = jet_lut_bgr[:, 1] / 255.0
        self.jet_lut_rgba[:, 2] = jet_lut_bgr[:, 0] / 255.0
        self.jet_lut_rgba[:, 3] = 1.0

        self.axes_overlay = PlanarAxesOverlay(self)
        self._sync_axes_overlay()

    def reset_camera_view(self):
        self.opts['fov'] = 60
        self.setCameraPosition(
            pos=QtGui.QVector3D(0, 0, 0),
            distance=800,
            elevation=40,
            azimuth=-90
        )

    def _sync_axes_overlay(self):
        overlay = getattr(self, 'axes_overlay', None)
        if overlay is None:
            return
        overlay.setGeometry(self.rect())
        overlay.raise_()
        if not overlay.isVisible():
            overlay.show()
        overlay.update()

    def _update_axes_overlay(self):
        overlay = getattr(self, 'axes_overlay', None)
        if overlay is not None:
            overlay.raise_()
            overlay.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_axes_overlay()

    def setCameraPosition(self, *args, **kwargs):
        super().setCameraPosition(*args, **kwargs)
        self._update_axes_overlay()

    def orbit(self, *args, **kwargs):
        super().orbit(*args, **kwargs)
        self._update_axes_overlay()

    def pan(self, *args, **kwargs):
        super().pan(*args, **kwargs)
        self._update_axes_overlay()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self._update_axes_overlay()

    def _build_cube_meshdata(self):
        base_half = self.profile_base_half
        z0, z2 = -0.5, 0.5
        corner_r = self.profile_corner_r
        corner_seg = self.profile_corner_seg
        fillet_z = self.profile_fillet_z
        z_steps = self.profile_z_steps

        def rounded_ring(half, radius, z):
            radius = max(1e-4, min(radius, half * 0.95))
            centers = [
                (half - radius, -half + radius, -np.pi / 2, 0.0),
                (half - radius, half - radius, 0.0, np.pi / 2),
                (-half + radius, half - radius, np.pi / 2, np.pi),
                (-half + radius, -half + radius, np.pi, 3 * np.pi / 2),
            ]
            pts = []
            for cx, cy, a0, a1 in centers:
                angles = np.linspace(a0, a1, corner_seg, endpoint=True)
                if pts:
                    angles = angles[1:]
                for a in angles:
                    pts.append((cx + radius * np.cos(a), cy + radius * np.sin(a), z))
            return pts

        rings = []
        for i in range(z_steps + 1):
            t = i / z_steps
            z = z0 + (z2 - z0) * t
            d_bot = max(0.0, (z0 + fillet_z) - z)
            d_top = max(0.0, z - (z2 - fillet_z))
            inset_bot = fillet_z - np.sqrt(max(fillet_z * fillet_z - d_bot * d_bot, 0.0))
            inset_top = fillet_z - np.sqrt(max(fillet_z * fillet_z - d_top * d_top, 0.0))
            inset = inset_bot + inset_top
            half = max(0.2, base_half - inset)
            rings.append(rounded_ring(half, corner_r, z))

        verts = np.array([v for ring in rings for v in ring], dtype=np.float32)
        faces = []
        n_per_ring = len(rings[0])

        for rid in range(len(rings) - 1):
            a0 = rid * n_per_ring
            b0 = (rid + 1) * n_per_ring
            for k in range(n_per_ring):
                n = (k + 1) % n_per_ring
                a = a0 + k
                an = a0 + n
                b = b0 + k
                bn = b0 + n
                faces.append([a, b, bn])
                faces.append([a, bn, an])

        bottom_center = len(verts)
        top_center = len(verts) + 1
        verts = np.vstack([
            verts,
            np.array([[0.0, 0.0, z0], [0.0, 0.0, z2]], dtype=np.float32)
        ])

        b0 = 0
        for k in range(n_per_ring):
            n = (k + 1) % n_per_ring
            faces.append([bottom_center, b0 + n, b0 + k])

        t0 = (len(rings) - 1) * n_per_ring
        for k in range(n_per_ring):
            n = (k + 1) % n_per_ring
            faces.append([top_center, t0 + k, t0 + n])

        return gl.MeshData(vertexes=verts, faces=np.asarray(faces, dtype=np.uint32))

    def _rounded_corner_px(self, w, h):
        r_ratio = min(self.profile_corner_r / max(self.profile_base_half, 1e-6), 0.95)
        return max(2.0, min(w, h) * 0.5 * r_ratio)

    def _is_point_in_rounded_rect(self, x, y, w, h, r):
        x = float(x)
        y = float(y)
        r = float(max(0.0, min(r, w * 0.5, h * 0.5)))
        if (r <= 1e-6) or (r <= x <= w - r) or (r <= y <= h - r):
            return (0.0 <= x <= w) and (0.0 <= y <= h)
        cx = r if x < r else (w - r)
        cy = r if y < r else (h - r)
        return ((x - cx) ** 2 + (y - cy) ** 2) <= (r ** 2)

    def _apply_rounded_alpha_mask(self, rgba_img, corner_px):
        h, w = rgba_img.shape[:2]
        cache_key = (h, w, int(corner_px))
        mask = self._rounded_mask_cache.get(cache_key)
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            r = int(max(1, min(corner_px, min(w, h) * 0.5 - 1)))
            cv2.rectangle(mask, (r, 0), (w - r, h), 255, thickness=-1)
            cv2.rectangle(mask, (0, r), (w, h - r), 255, thickness=-1)
            cv2.circle(mask, (r, r), r, 255, thickness=-1)
            cv2.circle(mask, (w - r, r), r, 255, thickness=-1)
            cv2.circle(mask, (r, h - r), r, 255, thickness=-1)
            cv2.circle(mask, (w - r, h - r), r, 255, thickness=-1)
            self._rounded_mask_cache[cache_key] = mask
        out = rgba_img.copy()
        out[..., 3] = (out[..., 3].astype(np.uint16) * mask.astype(np.uint16) // 255).astype(np.uint8)
        return out

    def _overlay_ui_scale(self):
        width_scale = max(self.width(), 1) / 1000.0
        height_scale = max(self.height(), 1) / 620.0
        return max(0.72, min(1.35, min(width_scale, height_scale)))

    def _draw_arrow_2d(self, painter, start_pt, end_pt, color, text):
        """绘制2D箭头（屏幕叠加层），与3D模型解耦。"""
        overlay_scale = self._overlay_ui_scale()
        pen = QPen(color, 2.6 * overlay_scale)
        painter.setPen(pen)
        painter.drawLine(start_pt, end_pt)

        vec = end_pt - start_pt
        vec_len = (vec.x() ** 2 + vec.y() ** 2) ** 0.5
        if vec_len < 1e-6:
            return

        ux, uy = vec.x() / vec_len, vec.y() / vec_len
        head_len = 8.5 * overlay_scale
        head_w = 4.5 * overlay_scale

        tip = end_pt
        left = QtCore.QPointF(
            tip.x() - ux * head_len - uy * head_w,
            tip.y() - uy * head_len + ux * head_w
        )
        right = QtCore.QPointF(
            tip.x() - ux * head_len + uy * head_w,
            tip.y() - uy * head_len - ux * head_w
        )

        painter.setBrush(color)
        painter.drawPolygon(tip, left, right)
        painter.drawText(QtCore.QPointF(tip.x() + 4 * overlay_scale, tip.y() - 4 * overlay_scale), text)

    def _draw_overlay_axes(self, painter):
        """左上角固定坐标系：在按键下方、脱离模型显示。"""
        overlay_scale = self._overlay_ui_scale()
        origin = QtCore.QPointF(50 * overlay_scale, 72 * overlay_scale)

        color_x = QColor(255, 84, 84)
        color_y = QColor(82, 232, 120)
        color_z = QColor(95, 148, 255)

        
        try:
            axis_len = 34.0 * overlay_scale
            world_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            world_axes_end = {
                'X (右)': np.array([1.0, 0.0, 0.0], dtype=np.float32),
                'Y (上)': np.array([0.787, 0.787, 0.0], dtype=np.float32),
                'Z (里)': np.array([0.0, -1.0, 0.0], dtype=np.float32),
            }

            axis_colors = {'X (右)': color_x, 'Y (上)': color_y, 'Z (里)': color_z}

            font = QFont("Microsoft YaHei UI")
            font.setPointSizeF(max(7.0, 10.0 * overlay_scale))
            font.setBold(True)
            painter.setFont(font)

            p0 = self._project_world_to_screen(world_origin)
            if p0 is None:
                return

            for name, pend_world in world_axes_end.items():
                p1 = self._project_world_to_screen(pend_world)
                if p1 is None:
                    continue

                d = p1 - p0
                norm2d = float(np.linalg.norm(d))
                if norm2d < 1e-6:
                    continue
                d = d / norm2d

                end = QtCore.QPointF(
                    origin.x() + float(d[0]) * axis_len,
                    origin.y() + float(d[1]) * axis_len
                )
                self._draw_arrow_2d(painter, origin, end, axis_colors[name], name)
        except Exception:
            
            font = QFont("Microsoft YaHei UI")
            font.setPointSizeF(max(7.0, 10.0 * overlay_scale))
            font.setBold(True)
            painter.setFont(font)
            self._draw_arrow_2d(painter, origin, QtCore.QPointF(origin.x() + 36 * overlay_scale, origin.y()), color_x, "X")
            self._draw_arrow_2d(painter, origin, QtCore.QPointF(origin.x(), origin.y() + 34 * overlay_scale), color_y, "Y")
            self._draw_arrow_2d(
                painter, origin,
                QtCore.QPointF(origin.x() + 24 * overlay_scale, origin.y() - 24 * overlay_scale),
                color_z, "Z")

    def _qmatrix_to_numpy(self, m):
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
        return np.array([x, y], dtype=np.float32)

    def paintEvent(self, event):
        super().paintEvent(event)
        self._update_axes_overlay()

    def set_board_image_bgr(self, bgr_img: np.ndarray, thickness: float = 24.0):
        if bgr_img is None or bgr_img.size == 0:
            return

        h, w = bgr_img.shape[:2]

        
        if self.board_h != h or self.board_w != w or self._cached_tw == 0:
            self.board_h, self.board_w = h, w
            self.board_thickness = float(thickness)

            board_scale = 1.10 * self.display_scale
            body_tr = pg.Transform3D()
            body_tr.scale(w * board_scale, h * board_scale, self.board_thickness)
            body_tr.translate(0, 0, -0.75)
            self.board_body.setTransform(body_tr)

            top_scale = board_scale * self.overlay_to_board_ratio
            self._cached_tw = max(2, int(round(w * top_scale)))
            self._cached_th = max(2, int(round(h * top_scale)))
            self._cached_corner_px = self._rounded_corner_px(self._cached_tw, self._cached_th)

            
            tr = pg.Transform3D()
            tr.translate(-self._cached_tw / 2, -self._cached_th / 2, self.overlay_z_offset)
            self.image_item.setTransform(tr)

        rgba = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGBA)
        rgba_top = cv2.resize(rgba, (self._cached_tw, self._cached_th), interpolation=cv2.INTER_LINEAR)
        rgba_top = self._apply_rounded_alpha_mask(rgba_top, self._cached_corner_px)

        self.image_item.setData(data=np.transpose(rgba_top, (1, 0, 2)))

    def _preallocate_planar_arrows(self, count):
        """预分配平面模式箭头对象池（只保留外轮廓线）"""
        while len(self.arrow_items) < count:
            item = gl.GLLinePlotItem(pos=np.empty((0, 3)), color=np.empty((0, 4)),
                                     width=1, mode='lines', antialias=False)
            item.setGLOptions('opaque')
            item.hide()
            self.addItem(item)
            self.arrow_items.append(item)
        while len(self.arrow_heads) < count:
            cone = gl.GLMeshItem(
                meshdata=self.arrow_head_mesh,
                color=(1, 1, 0, 0.95), smooth=False,
                drawEdges=False
            )
            cone.setGLOptions('translucent')
            cone.hide()
            self.addItem(cone)
            self.arrow_heads.append(cone)

    def _clear_arrow_heads(self):
        """隐藏所有箭头头部（复用，只隐藏）"""
        for cone in self.arrow_heads:
            cone.hide()

    def _clear_arrow_items(self):
        """隐藏所有箭头杆（复用，只隐藏）"""
        for item in self.arrow_items:
            item.hide()

    def _update_arrow_heads(self, arrow_vectors, arrow_colors, z_base, top_w, top_h):
        """更新箭头头部（复用对象池，只更新位置/旋转/缩放）"""
        
        self._clear_arrow_heads()
        if not arrow_vectors:
            return

        
        self._preallocate_planar_arrows(len(arrow_vectors))

        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for i, v in enumerate(arrow_vectors):
            sx, sy, ex, ey, dz = v
            dir_vec = np.array([ex - sx, ey - sy, dz], dtype=np.float32)
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-6:
                continue
            direction = dir_vec / norm
            head_color = arrow_colors[i] if i < len(arrow_colors) else (1.0, 1.0, 0.0, 0.95)

            cone = self.arrow_heads[i]

            rotation = pg.Transform3D()
            if np.allclose(direction, z_axis):
                pass
            elif np.allclose(direction, -z_axis):
                rotation.rotate(180, 1, 0, 0)
            else:
                axis = np.cross(z_axis, direction)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    angle = np.degrees(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
                    rotation.rotate(angle, axis[0], axis[1], axis[2])

            t = pg.Transform3D()
            t.translate(ex - top_w / 2, ey - top_h / 2, z_base + dz)
            t = t * rotation
            
            head_scale = 0.3 + min(norm * 0.05, 8.2)
            t.scale(head_scale, head_scale, head_scale)
            cone.setTransform(t)
            cone.setColor(head_color)
            cone.show()

    def set_arrow_vectors(self, arrow_vectors, colors_rgba):
        """批量渲染所有箭头：1 个 GLLinePlotItem 画全部箭杆 + 1 个 InstancedConeItem 画全部箭头（GPU 实例化）。
        与指尖模式 FingertipBoardGLWidget 采用相同的优化策略，将 512 次 OpenGL 调用合并为 2 次。"""
        self.batched_arrow_lines.hide()
        self.batched_arrow_heads_item.hide()

        if not arrow_vectors:
            return

        if self.board_w <= 0 or self.board_h <= 0:
            return

        board_scale = 1.10 * self.display_scale
        top_scale = board_scale * self.overlay_to_board_ratio
        top_w = self.board_w * top_scale
        top_h = self.board_h * top_scale
        corner_px = self._rounded_corner_px(top_w, top_h)
        z_base = self.arrow_z_base

        line_pts_list = []
        line_colors_list = []
        head_positions = []
        head_directions = []
        head_scales = []
        head_colors_list = []

        for i, (sx, sy, ex, ey, dz) in enumerate(arrow_vectors):
            sx2 = (sx - self.board_w / 2.0) * top_scale + top_w / 2.0
            sy2 = (sy - self.board_h / 2.0) * top_scale + top_h / 2.0
            ex2 = (ex - self.board_w / 2.0) * top_scale + top_w / 2.0
            ey2 = (ey - self.board_h / 2.0) * top_scale + top_h / 2.0

            if not self._is_point_in_rounded_rect(sx2, sy2, top_w, top_h, corner_px):
                continue
            if not self._is_point_in_rounded_rect(ex2, ey2, top_w, top_h, corner_px):
                continue

            c0 = colors_rgba[i * 2] if (i * 2) < len(colors_rgba) else (1.0, 1.0, 0.0, 1.0)
            c1 = colors_rgba[i * 2 + 1] if (i * 2 + 1) < len(colors_rgba) else c0

            start_pt = np.array([sx2 - top_w / 2, sy2 - top_h / 2, z_base], dtype=np.float32)
            end_pt = np.array([ex2 - top_w / 2, ey2 - top_h / 2, z_base + dz], dtype=np.float32)

            line_pts_list.append(start_pt)
            line_pts_list.append(end_pt)
            line_colors_list.append(tuple(float(c) for c in c0))
            line_colors_list.append(tuple(float(c) for c in c1))

            
            vec = end_pt - start_pt
            fn = float(np.linalg.norm(vec))
            if fn > 0.01:
                head_positions.append(end_pt)
                head_directions.append(vec / fn)
                head_scales.append(0.5 + min(fn * 0.05, 8.2))
                head_colors_list.append(tuple(float(c) for c in c0))

        
        if line_pts_list:
            line_pts = np.array(line_pts_list, dtype=np.float32)
            line_colors = np.array(line_colors_list, dtype=np.float32)
            self.batched_arrow_lines.setData(pos=line_pts, color=line_colors, width=4.0, mode='lines')
            self.batched_arrow_lines.show()

        
        if head_positions:
            self.batched_arrow_heads_item.set_instances(
                np.array(head_positions, dtype=np.float32),
                np.array(head_directions, dtype=np.float32),
                np.array(head_scales, dtype=np.float32),
                np.array(head_colors_list, dtype=np.float32)
            )




DEVICE_SLIDER_CONFIG = {
    'bevel': {
        'heatmap':  {'range': (10, 2000), 'default': 500},    
        'arrow_xy': {'range': (10, 800),  'default': 400},    
        'arrow_z':  {'range': (10, 300),  'default': 150},    
    },
    'curved': {
        'heatmap':  {'range': (10, 6000), 'default': 800},   
        'arrow_xy': {'range': (10, 1500), 'default': 400},    
        'arrow_z':  {'range': (10, 600),  'default': 200},    
    },
}


class Force3DSurfaceApp(QtWidgets.QWidget):
    BASE_WINDOW_W = 1440
    BASE_WINDOW_H = 950

    def _current_screen(self):
        screen_at = getattr(QtWidgets.QApplication, 'screenAt', None)
        if screen_at is not None:
            screen = screen_at(QtGui.QCursor.pos())
            if screen is not None:
                return screen
        return QtWidgets.QApplication.primaryScreen()

    def _compute_screen_layout(self):
        screen = self._current_screen()
        if screen is None:
            return 1.0, QtCore.QSize(self.BASE_WINDOW_W, self.BASE_WINDOW_H)

        available = screen.availableGeometry()
        usable_w = max(800, int(available.width() * 0.92))
        usable_h = max(560, int(available.height() * 0.92))
        scale = min(usable_w / self.BASE_WINDOW_W, usable_h / self.BASE_WINDOW_H)
        scale = max(0.72, min(1.15, scale))
        width = min(usable_w, max(960, int(round(self.BASE_WINDOW_W * scale))))
        height = min(usable_h, max(620, int(round(self.BASE_WINDOW_H * scale))))
        return scale, QtCore.QSize(width, height)

    def _center_on_current_screen(self):
        screen = self._current_screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        rect = QtCore.QRect(QtCore.QPoint(0, 0), self.size())
        rect.moveCenter(available.center())
        self.move(rect.topLeft())

    def _px(self, value):
        return max(1, int(round(float(value) * self.ui_scale)))

    def _pt(self, value):
        return max(7.0, float(value) * self.ui_scale)

    def __init__(self, sensor=None, max_hist_time=3):
        super().__init__()
        self._gui_rates = FrameRateCounter()
        self._shutdown_complete = False
        self.sensor = sensor
        self.device_type = getattr(sensor, 'device_type', None)
        self.current_port = None
        self.setWindowTitle("3D压力表面 + 切向力场 + 力分析")
        self.ui_scale, initial_size = self._compute_screen_layout()
        self.resize(initial_size)
        self._center_on_current_screen()
        self.setMinimumSize(self._px(980), self._px(640))
        self.setObjectName("mainWindow")
        ui_font = QFont("Microsoft YaHei UI")
        ui_font.setPointSizeF(self._pt(10))
        self.setFont(ui_font)
        self.setStyleSheet("""
            QWidget#mainWindow {
                background-color: #000000;
                color: #e5e7eb;
                font-family: "Microsoft YaHei UI", "Segoe UI", "PingFang SC", "Noto Sans CJK SC";
                font-size: %.1fpt;
            }
            QPushButton {
                color: #ffffff;
                font-weight: 600;
                font-size: %.1fpt;
                font-family: "Microsoft YaHei UI", "Segoe UI", "PingFang SC", "Noto Sans CJK SC";
                padding: %dpx %dpx;
                border-radius: %dpx;
                border: 1px solid rgba(255, 255, 255, 90);
                background-color: rgba(255, 255, 255, 45);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 70);
                border: 1px solid rgba(255, 255, 255, 130);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 95);
                border: 1px solid rgba(255, 255, 255, 160);
            }
            QFrame#glassPanel {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.00 rgba(255, 255, 255, 46),
                    stop: 0.08 rgba(255, 255, 255, 26),
                    stop: 0.20 rgba(12, 16, 24, 165),
                    stop: 1.00 rgba(12, 16, 24, 165)
                );
                border: 1px solid rgba(255, 255, 255, 72);
                border-radius: %dpx;
            }
            QLabel {
                font-family: "Microsoft YaHei UI", "Segoe UI", "PingFang SC", "Noto Sans CJK SC";
                font-size: %.1fpt;
            }
        """ % (
            self._pt(10.5), self._pt(11),
            self._px(4), self._px(10), self._px(6),
            self._px(14), self._pt(10.5)
        ))

        main_vbox = QtWidgets.QVBoxLayout()
        main_vbox.setContentsMargins(self._px(9), self._px(9), self._px(9), self._px(9))
        main_vbox.setSpacing(self._px(4))
        self.setLayout(main_vbox)

        
        upper_hbox = QtWidgets.QHBoxLayout()

        self.pressure_stack = QtWidgets.QStackedWidget()
        self.planar_gl_widget = PlanarBoardGLWidget()
        self.pressure_stack.addWidget(self.planar_gl_widget)  

        self.fingertip_gl_widget = FingertipBoardGLWidget()
        self.pressure_stack.addWidget(self.fingertip_gl_widget)  

        
        self.radar_panel = QtWidgets.QFrame()
        self.radar_panel.setObjectName("glassPanel")
        self.radar_panel.setMinimumWidth(self._px(400))
        self.radar_panel.setMaximumWidth(self._px(430))

        radar_panel_layout = QtWidgets.QVBoxLayout(self.radar_panel)
        radar_panel_layout.setContentsMargins(self._px(10), self._px(8), self._px(10), self._px(8))
        radar_panel_layout.setSpacing(self._px(4))

        self.radar_panel_title = QtWidgets.QLabel("雷达图")
        self.radar_panel_title.setStyleSheet(
            f"color:#e5e7eb; font-size:{self._pt(13):.1f}pt; font-weight:700;")
        radar_panel_layout.addWidget(self.radar_panel_title)

        self.radar_plot = RadarPlotWidget(
            RadarComputeWorker.RADAR_LABELS,
            signed_feature_indices=RadarComputeWorker.SIGNED_FEATURE_INDICES,
            signed_feature_deadbands=(
                RadarComputeWorker.SIGNED_FEATURE_DEADBANDS),
            label_font_size=7,
        )
        self.radar_plot.setMinimumHeight(self._px(300))
        radar_panel_layout.addWidget(self.radar_plot, stretch=1)

        self.radar_status_label = QtWidgets.QLabel("状态: 等待数据")
        self.radar_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.radar_status_label.setStyleSheet(
            f"color:#9fb0c8; font-size:{self._pt(10):.1f}pt; font-weight:600;")
        radar_panel_layout.addWidget(self.radar_status_label)

        
        self.force_info_panel = QtWidgets.QFrame()
        self.force_info_panel.setObjectName("glassPanel")
        self.force_info_panel.setMinimumWidth(self._px(260))
        self.force_info_panel.setMaximumWidth(self._px(360))

        force_info_layout = QtWidgets.QVBoxLayout(self.force_info_panel)
        force_info_layout.setContentsMargins(self._px(10), self._px(8), self._px(10), self._px(8))
        force_info_layout.setSpacing(self._px(4))

        self.force_info_title = QtWidgets.QLabel("三轴力")
        self.force_info_title.setStyleSheet(
            f"color:#e5e7eb; font-size:{self._pt(13):.1f}pt; font-weight:700;")
        force_info_layout.addWidget(self.force_info_title)

        self.fx_value_label = QtWidgets.QLabel("Fx: 0 mN")
        self.fy_value_label = QtWidgets.QLabel("Fy: 0 mN")
        self.fz_value_label = QtWidgets.QLabel("Fz: 0 mN")

        for lbl in [self.fx_value_label, self.fy_value_label, self.fz_value_label]:
            lbl.setStyleSheet(f"color:#d7dfeb; font-size:{self._pt(11.5):.1f}pt; font-weight:600;")
            force_info_layout.addWidget(lbl)

        
        self.heatmap_scale = 80
        self.heatmap_scale_title = QtWidgets.QLabel("热力图缩放")
        self.heatmap_scale_title.setStyleSheet(
            f"color:#e5e7eb; font-size:{self._pt(10.5):.1f}pt; font-weight:650;")
        force_info_layout.addWidget(self.heatmap_scale_title)

        scale_slider_layout = QtWidgets.QHBoxLayout()
        self.heatmap_scale_label = QtWidgets.QLabel(f"{self.heatmap_scale:.0f}")
        self.heatmap_scale_label.setStyleSheet(
            f"color:#d7dfeb; font-size:{self._pt(9.5):.1f}pt; "
            f"font-weight:600; min-width:{self._px(30)}px;")
        self.heatmap_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.heatmap_scale_slider.setRange(10, 1000)
        self.heatmap_scale_slider.valueChanged.connect(self._on_heatmap_scale_changed)
        self.heatmap_scale_slider.setValue(int(self.heatmap_scale * 10))
        self.heatmap_scale_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: %dpx;
                background: #3a4050;
                border-radius: %dpx;
            }
            QSlider::handle:horizontal {
                width: %dpx;
                height: %dpx;
                margin: -%dpx 0;
                background: #7ba0ff;
                border-radius: %dpx;
            }
            QSlider::sub-page:horizontal {
                background: #7ba0ff;
                border-radius: %dpx;
            }
        """ % (
            self._px(3), self._px(1),
            self._px(12), self._px(12), self._px(4), self._px(6),
            self._px(1)
        ))
        
        btn_style = (
            f"font-size:{self._pt(10)}pt; font-weight:700;"
            f"padding:0px; min-width:{self._px(20)}px; min-height:{self._px(20)}px;"
            "background: transparent; border: none;"
        )
        self.scale_minus_btn = QtWidgets.QPushButton("−")
        self.scale_minus_btn.setFixedSize(self._px(20), self._px(20))
        self.scale_minus_btn.setStyleSheet(btn_style)
        self.scale_minus_btn.clicked.connect(self._on_scale_minus)
        self.scale_plus_btn = QtWidgets.QPushButton("+")
        self.scale_plus_btn.setFixedSize(self._px(20), self._px(20))
        self.scale_plus_btn.setStyleSheet(btn_style)
        self.scale_plus_btn.clicked.connect(self._on_scale_plus)

        scale_slider_layout.addWidget(self.scale_minus_btn)
        scale_slider_layout.addWidget(self.heatmap_scale_label)
        scale_slider_layout.addWidget(self.scale_plus_btn)
        scale_slider_layout.addWidget(self.heatmap_scale_slider)
        force_info_layout.addLayout(scale_slider_layout)

        
        self.arrow_xy_scale = 40  
        self.arrow_scale_title = QtWidgets.QLabel("箭头XY比例")
        self.arrow_scale_title.setStyleSheet(
            f"color:#e5e7eb; font-size:{self._pt(10.5):.1f}pt; font-weight:650;")
        force_info_layout.addWidget(self.arrow_scale_title)

        arrow_scale_slider_layout = QtWidgets.QHBoxLayout()
        self.arrow_scale_label = QtWidgets.QLabel(f"{self.arrow_xy_scale:.0f}")
        self.arrow_scale_label.setStyleSheet(
            f"color:#d7dfeb; font-size:{self._pt(9.5):.1f}pt; "
            f"font-weight:600; min-width:{self._px(30)}px;")
        self.arrow_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.arrow_scale_slider.setRange(10, 10000)
        self.arrow_scale_slider.valueChanged.connect(self._on_arrow_scale_changed)
        self.arrow_scale_slider.setValue(int(self.arrow_xy_scale * 10))
        self.arrow_scale_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: %dpx;
                background: #3a4050;
                border-radius: %dpx;
            }
            QSlider::handle:horizontal {
                width: %dpx;
                height: %dpx;
                margin: -%dpx 0;
                background: #7ba0ff;
                border-radius: %dpx;
            }
            QSlider::sub-page:horizontal {
                background: #7ba0ff;
                border-radius: %dpx;
            }
        """ % (
            self._px(3), self._px(1),
            self._px(12), self._px(12), self._px(4), self._px(6),
            self._px(1)
        ))
        self.arrow_scale_minus_btn = QtWidgets.QPushButton("−")
        self.arrow_scale_minus_btn.setFixedSize(self._px(20), self._px(20))
        self.arrow_scale_minus_btn.setStyleSheet(btn_style)
        self.arrow_scale_minus_btn.clicked.connect(self._on_arrow_scale_minus)
        self.arrow_scale_plus_btn = QtWidgets.QPushButton("+")
        self.arrow_scale_plus_btn.setFixedSize(self._px(20), self._px(20))
        self.arrow_scale_plus_btn.setStyleSheet(btn_style)
        self.arrow_scale_plus_btn.clicked.connect(self._on_arrow_scale_plus)

        arrow_scale_slider_layout.addWidget(self.arrow_scale_minus_btn)
        arrow_scale_slider_layout.addWidget(self.arrow_scale_label)
        arrow_scale_slider_layout.addWidget(self.arrow_scale_plus_btn)
        arrow_scale_slider_layout.addWidget(self.arrow_scale_slider)
        force_info_layout.addLayout(arrow_scale_slider_layout)

        
        self.arrow_z_scale = 20  
        self.arrow_z_scale_title = QtWidgets.QLabel("箭头Z比例")
        self.arrow_z_scale_title.setStyleSheet(
            f"color:#e5e7eb; font-size:{self._pt(10):.1f}pt; font-weight:650;")
        force_info_layout.addWidget(self.arrow_z_scale_title)

        arrow_z_scale_slider_layout = QtWidgets.QHBoxLayout()
        self.arrow_z_scale_label = QtWidgets.QLabel(f"{self.arrow_z_scale:.0f}")
        self.arrow_z_scale_label.setStyleSheet(
            f"color:#d7dfeb; font-size:{self._pt(9.5):.1f}pt; "
            f"font-weight:600; min-width:{self._px(30)}px;")
        self.arrow_z_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.arrow_z_scale_slider.setRange(10, 2000)
        self.arrow_z_scale_slider.valueChanged.connect(self._on_arrow_z_scale_changed)
        self.arrow_z_scale_slider.setValue(int(self.arrow_z_scale * 10))
        self.arrow_z_scale_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: %dpx;
                background: #3a4050;
                border-radius: %dpx;
            }
            QSlider::handle:horizontal {
                width: %dpx;
                height: %dpx;
                margin: -%dpx 0;
                background: #7ba0ff;
                border-radius: %dpx;
            }
            QSlider::sub-page:horizontal {
                background: #7ba0ff;
                border-radius: %dpx;
            }
        """ % (
            self._px(3), self._px(1),
            self._px(12), self._px(12), self._px(4), self._px(6),
            self._px(1)
        ))
        self.arrow_z_scale_minus_btn = QtWidgets.QPushButton("−")
        self.arrow_z_scale_minus_btn.setFixedSize(self._px(20), self._px(20))
        self.arrow_z_scale_minus_btn.setStyleSheet(btn_style)
        self.arrow_z_scale_minus_btn.clicked.connect(self._on_arrow_z_scale_minus)
        self.arrow_z_scale_plus_btn = QtWidgets.QPushButton("+")
        self.arrow_z_scale_plus_btn.setFixedSize(self._px(20), self._px(20))
        self.arrow_z_scale_plus_btn.setStyleSheet(btn_style)
        self.arrow_z_scale_plus_btn.clicked.connect(self._on_arrow_z_scale_plus)

        arrow_z_scale_slider_layout.addWidget(self.arrow_z_scale_minus_btn)
        arrow_z_scale_slider_layout.addWidget(self.arrow_z_scale_label)
        arrow_z_scale_slider_layout.addWidget(self.arrow_z_scale_plus_btn)
        arrow_z_scale_slider_layout.addWidget(self.arrow_z_scale_slider)
        force_info_layout.addLayout(arrow_z_scale_slider_layout)

        
        self.reset_sliders_btn = QtWidgets.QPushButton("恢复默认值")
        self.reset_sliders_btn.setFixedHeight(self._px(30))
        self.reset_sliders_btn.setStyleSheet(
            f"font-size:{self._pt(10)}pt; font-weight:600;"
            f"color:#d7dfeb; padding:{self._px(3)}px {self._px(8)}px;"
            "background: rgba(255,255,255,10);"
            f"border: 1px solid rgba(255,255,255,60);"
            f"border-radius: {self._px(5)}px;"
        )
        self.reset_sliders_btn.clicked.connect(self._on_reset_slider_defaults)
        force_info_layout.addWidget(self.reset_sliders_btn)

        self.side_map_title = QtWidgets.QLabel("平面热力图")

        self.side_map_title.setStyleSheet(
            f"color:#e5e7eb; font-size:{self._pt(10.5):.1f}pt; font-weight:650;")
        force_info_layout.addWidget(self.side_map_title)

        self.side_map_label = QtWidgets.QLabel("无数据")
        self.side_map_label.setAlignment(QtCore.Qt.AlignCenter)
        self.side_map_label.setMinimumHeight(self._px(170))
        self.side_map_label.setStyleSheet(
            "background-color: rgba(0,0,0,80);"
            "border: 1px solid rgba(255,255,255,60);"
            f"border-radius: {self._px(8)}px;"
            "color:#b7c3d6;"
        )
        force_info_layout.addWidget(self.side_map_label)

        
        self._version_font_size = self._pt(8)
        self.gui_version_label = QtWidgets.QLabel(f"GUI版本: {GUI_VERSION}")
        self.gui_version_label.setStyleSheet(
            f"color:#6b7280; font-size:{self._version_font_size:.1f}pt; font-weight:400;")
        force_info_layout.addWidget(self.gui_version_label)

        self.api_version_label = QtWidgets.QLabel(f"API版本: {API_VERSION}")
        self.api_version_label.setStyleSheet(
            f"color:#6b7280; font-size:{self._version_font_size:.1f}pt; font-weight:400;")
        force_info_layout.addWidget(self.api_version_label)

        force_info_layout.addStretch(1)
        
        upper_hbox.addWidget(self.pressure_stack, stretch=4) 
        upper_hbox.addWidget(self.radar_panel, stretch=1)
        upper_hbox.addWidget(self.force_info_panel, stretch=1)

        
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(self._px(6))
        self.calibrate_btn = QtWidgets.QPushButton("校准 (C键)")
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.calibrate_btn.setFixedHeight(self._px(31))
        button_layout.addWidget(self.calibrate_btn)

        self.autocal_toggle_btn = QtWidgets.QPushButton("自动校准：开启")
        self.autocal_toggle_btn.clicked.connect(self.toggle_auto_calibration)
        self.autocal_toggle_btn.setFixedHeight(self._px(31))
        button_layout.addWidget(self.autocal_toggle_btn)

        self.switch_device_btn = QtWidgets.QPushButton("更换设备")
        self.switch_device_btn.clicked.connect(self.back_to_port_selection)
        self.switch_device_btn.setFixedHeight(self._px(31))
        button_layout.addWidget(self.switch_device_btn)

        
        align_layout = QtWidgets.QHBoxLayout()
        align_layout.setSpacing(self._px(6))
        self.dynamic_range_enabled = False
        self.dynamic_range_btn = QtWidgets.QPushButton("Y轴自适应：关闭")
        self.dynamic_range_btn.clicked.connect(self.toggle_dynamic_range)
        self.dynamic_range_btn.setFixedHeight(self._px(31))

        self.reset_view_btn = QtWidgets.QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setFixedHeight(self._px(31))

        align_layout.addWidget(self.dynamic_range_btn)
        align_layout.addWidget(self.reset_view_btn)

        

        self.device_status_label = QtWidgets.QLabel("设备状态：正在等待连接")
        self.device_status_label.setWordWrap(True)
        self.device_status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.device_status_label.setMinimumHeight(self._px(42))

        main_vbox.addLayout(button_layout)
        main_vbox.addLayout(align_layout)
        main_vbox.addWidget(self.device_status_label)
        main_vbox.addLayout(upper_hbox, stretch=3)

        
        self.glass_panel = QtWidgets.QFrame()
        self.glass_panel.setObjectName("glassPanel")
        lower_plots_hbox = QtWidgets.QHBoxLayout(self.glass_panel)
        lower_plots_hbox.setContentsMargins(self._px(12), self._px(10), self._px(12), self._px(10))
        lower_plots_hbox.setSpacing(self._px(10))
        self.max_hist_time = float(max_hist_time)
        self.render_target_fps = 30
        self.hist_len = max(2, int(round(self.max_hist_time * self.render_target_fps)))
        self.fx_hist = np.zeros(self.hist_len, dtype=np.float32)
        self.fy_hist = np.zeros(self.hist_len, dtype=np.float32)
        self.fz_hist = np.zeros(self.hist_len, dtype=np.float32)
        self.xy_smooth_alpha = 0.55
        self.fx_smooth_value = 0.0
        self.fy_smooth_value = 0.0
        self.idx = 0
        self.time_axis = np.linspace(-self.max_hist_time, 0.0, self.hist_len, dtype=np.float32)
        self.fz_fixed_y_min = -200.0
        self.fz_fixed_y_max = 2000.0
        self.fz_auto_y_min = -200.0
        self.fz_auto_min_span = 400.0

        
        axis_tick_font = QFont("Microsoft YaHei UI")
        axis_tick_font.setPointSizeF(self._pt(9))
        axis_line_pen = pg.mkPen('#93a1b5')
        axis_text_pen = pg.mkPen('#c8d2df')
        self.pgwin_tangent = pg.PlotWidget(
            title=(
                "<span style='color:#e5e7eb; font-family:Microsoft YaHei UI,Segoe UI,PingFang SC; "
                f"font-size:{self._pt(13):.1f}pt; font-weight:600;'>Fx / Fy</span>"
            ))
        self.pgwin_tangent.setBackground((0, 0, 0, 0))
        self.pgwin_tangent.setLabel('left', 'Force (mN)', **{
            'color': '#d7dfeb', 'size': f'{self._pt(11):.1f}pt',
            'font-family': 'Microsoft YaHei UI, Segoe UI, PingFang SC'})
        self.pgwin_tangent.setLabel('bottom', '时间 (s)', **{
            'color': '#d7dfeb', 'size': f'{self._pt(10):.1f}pt',
            'font-family': 'Microsoft YaHei UI, Segoe UI, PingFang SC'})
        self.pgwin_tangent.addLegend()
        self.pgwin_tangent.showGrid(x=False, y=False)
        for axis_name in ('left', 'bottom'):
            axis = self.pgwin_tangent.getAxis(axis_name)
            axis.setPen(axis_line_pen)
            axis.setTextPen(axis_text_pen)
            axis.setTickFont(axis_tick_font)
            axis.setStyle(tickFont=axis_tick_font, showValues=(axis_name != 'bottom'))
        self.pgwin_tangent.setStyleSheet("background: transparent; border: none;")
        self.pgwin_tangent.setXRange(float(self.time_axis[0]), 0.0, padding=0)
        self.pgwin_tangent.setYRange(-2000, 2000)
        self.pgwin_tangent.getViewBox().setMouseEnabled(x=False, y=False)
        self.pgwin_tangent.setMenuEnabled(False)
        self.fx_curve = self.pgwin_tangent.plot(
            self.time_axis, self.fx_hist,
            pen=pg.mkPen('#62D0FF', width=2.8),
            name='Fx'
        )
        self.fy_curve = self.pgwin_tangent.plot(
            self.time_axis, self.fy_hist,
            pen=pg.mkPen('#F3B56B', width=2.8),
            name='Fy'
        )
        lower_plots_hbox.addWidget(self.pgwin_tangent)

        
        self.pgwin_normal = pg.PlotWidget(
            title=(
                "<span style='color:#e5e7eb; font-family:Microsoft YaHei UI,Segoe UI,PingFang SC; "
                f"font-size:{self._pt(13):.1f}pt; font-weight:600;'>Fz</span>"
            ))
        self.pgwin_normal.setBackground((0, 0, 0, 0))
        self.pgwin_normal.showAxis('left')
        self.pgwin_normal.hideAxis('right')
        self.pgwin_normal.setLabel('left', 'Force (mN)', **{
            'color': '#d7dfeb', 'size': f'{self._pt(11):.1f}pt',
            'font-family': 'Microsoft YaHei UI, Segoe UI, PingFang SC'})
        self.pgwin_normal.setLabel('bottom', '时间 (s)', **{
            'color': '#d7dfeb', 'size': f'{self._pt(10):.1f}pt',
            'font-family': 'Microsoft YaHei UI, Segoe UI, PingFang SC'})
        self.pgwin_normal.addLegend()
        self.pgwin_normal.showGrid(x=False, y=False)
        for axis_name in ('left', 'bottom'):
            axis = self.pgwin_normal.getAxis(axis_name)
            axis.setPen(axis_line_pen)
            axis.setTextPen(axis_text_pen)
            axis.setTickFont(axis_tick_font)
            axis.setStyle(tickFont=axis_tick_font, showValues=(axis_name != 'bottom'))
        self.pgwin_normal.setStyleSheet("background: transparent; border: none;")
        self.pgwin_normal.setXRange(float(self.time_axis[0]), 0.0, padding=0)
        self.pgwin_normal.setYRange(self.fz_fixed_y_min, self.fz_fixed_y_max, padding=0)
        self.pgwin_normal.getViewBox().setMouseEnabled(x=False, y=False)
        self.pgwin_normal.setMenuEnabled(False)
        self.fz_curve = self.pgwin_normal.plot(
            self.time_axis, self.fz_hist,
            pen=pg.mkPen('#A78BFA', width=3.0),
            name='Fz'
        )
        lower_plots_hbox.addWidget(self.pgwin_normal)

        main_vbox.addWidget(self.glass_panel, stretch=2)

        
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.keyPressEvent = self.handle_key_press
        self.should_exit = False
        self.button_processing = False
        
        self.arrow_xy_scale = 100.0  
        self.arrow_z_scale = 15.0  
        self.arrow_color_gain = 400  
        self.arrow_min_force = 0.1  
        self.arrow_min_len = 4.0  

        
        self.flow_update_interval = 2
        self.flow_update_counter = 0
        self.cached_flow_x = None
        self.cached_flow_y = None
        self.tangent_stride = 2
        self.arrow_update_interval = 1  
        self.arrow_update_counter = 0
        self.cached_arrow_segments_2d = []
        self.cached_arrow_colors_rgba = []
        self._planar_arrow_visual_key = None
        self._planar_arrow_cache_key = None
        self._planar_arrow_cache = None
        self._current_pressure_index = 1 if self.device_type == 'curved' else 0
        self.pressure_stack.setCurrentIndex(self._current_pressure_index)


        self._sensor_worker = None

        
        self._radar_generation = 0
        self._radar_feature_worker = RadarFeatureWorker()
        self._radar_feature_worker.result_available.connect(
            self._consume_radar_result, QtCore.Qt.QueuedConnection)
        self._radar_feature_worker.start()

        self._startup_scan_worker = None
        self._startup_connect_request_id = 0
        self._sensor_init_worker = None
        
        
        self._manual_connect_pending_args = None

        self._apply_device_slider_config(self.device_type)
        self._reset_runtime_metrics()
        self._set_sensor_controls_enabled(self.sensor is not None)
        if self.sensor is not None:
            self._set_device_status("设备状态：设备已连接", "success")
            self.start_sensor_worker()
        else:
            self._set_device_status("设备状态：正在自动连接设备...", "warning")

    def _set_device_status(self, text, level="warning"):
        styles = {
            "warning": ("#f59e0b", "rgba(245, 158, 11, 38)", "rgba(245, 158, 11, 120)"),
            "success": ("#34d399", "rgba(52, 211, 153, 36)", "rgba(52, 211, 153, 110)"),
            "error": ("#f87171", "rgba(248, 113, 113, 36)", "rgba(248, 113, 113, 110)"),
        }
        color, bg, border = styles.get(level, styles["warning"])
        self.device_status_label.setText(text)
        self.device_status_label.setStyleSheet(
            "QLabel {"
            f"color: {color};"
            f"background-color: {bg};"
            f"border: 1px solid {border};"
            f"border-radius: {self._px(8)}px;"
            f"padding: {self._px(8)}px {self._px(12)}px;"
            f"font-size: {self._pt(10.2):.1f}pt;"
            "font-weight: 600;"
            "}"
        )

    def _set_sensor_controls_enabled(self, connected):
        self.calibrate_btn.setEnabled(bool(connected))
        self.autocal_toggle_btn.setEnabled(bool(connected))

    def _reset_runtime_metrics(self, side_map_text="正在等待设备数据..."):
        self._reset_rate_windows()
        self.fx_value_label.setText("Fx: -- mN")
        self.fy_value_label.setText("Fy: -- mN")
        self.fz_value_label.setText("Fz: -- mN")
        self.side_map_label.clear()
        self.side_map_label.setText(side_map_text)
        self.radar_plot.clear_radar()
        self.radar_status_label.setText("状态: 等待数据")
        self.fx_hist.fill(0.0)
        self.fy_hist.fill(0.0)
        self.fz_hist.fill(0.0)
        self.fx_curve.setData(self.time_axis, self.fx_hist)
        self.fy_curve.setData(self.time_axis, self.fy_hist)
        self.fz_curve.setData(self.time_axis, self.fz_hist)
        self.idx = 0
    def _invalidate_startup_auto_connect(self):
        self._startup_connect_request_id += 1
        return self._startup_connect_request_id

    def _reset_rate_windows(self):
        rates = getattr(self, '_gui_rates', None)
        if rates is not None:
            rates.reset()

    def start_sensor_worker(self):
        if self.sensor is None:
            return
        if self._sensor_worker is not None and self._sensor_worker.isRunning():
            return
        self._reset_rate_windows()
        self._sensor_worker = SensorUpdateWorker(self.sensor)
        self._sensor_worker.frame_available.connect(
            self._consume_sensor_frame, QtCore.Qt.QueuedConnection)
        self._sensor_worker.error_occurred.connect(self._on_worker_error, QtCore.Qt.QueuedConnection)
        self._sensor_worker.start()

    def stop_sensor_worker(self, timeout=2000):
        worker = self._sensor_worker
        self._sensor_worker = None
        if worker is None:
            return True
        worker.close_mailbox()
        if not worker.isRunning():
            return True
        worker.requestInterruption()
        if not worker.wait(timeout):
            print("传感器线程未能在超时前停止")
            worker.terminate()
            worker.wait()
            return False
        return True

    def begin_startup_auto_connect(self):
        if self._startup_scan_worker is not None and self._startup_scan_worker.isRunning():
            return

        request_id = self._invalidate_startup_auto_connect()
        self._reset_runtime_metrics("正在自动扫描并连接设备...")
        self._set_sensor_controls_enabled(False)
        self._set_device_status("设备状态：正在自动扫描并连接设备...", "warning")

        worker = StartupPortScanWorker(request_id)
        worker.scan_finished.connect(self._on_startup_scan_finished, QtCore.Qt.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        self._startup_scan_worker = worker
        worker.start()

    def _apply_connected_sensor(self, sensor_obj, selected_port=None):
        device_type = sensor_obj.device_type
        device_label, mode_label = _resolve_device_profile(device_type)

        self.sensor = sensor_obj
        self.current_port = selected_port
        self.device_type = device_type

        print(f"检测到{device_label}设备 (device_type={device_type!r})，自动切换{mode_label}模式")

        target_index = 1 if device_type == 'curved' else 0
        self.pressure_stack.setCurrentIndex(target_index)
        self._current_pressure_index = target_index

        self._apply_device_slider_config(device_type)
        self._reset_runtime_metrics("正在接收设备数据...")
        self._reset_radar_state()
        self.start_sensor_worker()
        self.fingertip_gl_widget.reset_view_data()
        self.should_exit = False
        self._set_sensor_controls_enabled(True)
        self._sync_autocal_toggle()

        if selected_port is None:
            self._set_device_status(f"设备状态：已连接（{device_label}）", "success")
        else:
            self._set_device_status(
                f"设备状态：已连接 USB端口 {selected_port}（{device_label}）",
                "success"
            )

    def _handle_startup_connect_failure(self, detail=None):
        self.sensor = None
        self.current_port = None
        self._set_sensor_controls_enabled(False)
        self._reset_runtime_metrics("自动连接失败！\n请点击“更换设备”手动连接。")
        if detail:
            print(f"启动自动连接失败: {detail}")
        self._set_device_status(
            "设备状态：自动连接失败，请点击“更换设备”手动连接。",
            "error"
        )

    @QtCore.pyqtSlot(int, object, str)
    def _on_startup_scan_finished(self, request_id, available_ports, error_msg):
        worker = self._startup_scan_worker
        if worker is not None and worker.request_id == request_id:
            self._startup_scan_worker = None

        if request_id != self._startup_connect_request_id:
            return

        if error_msg:
            self._handle_startup_connect_failure(error_msg)
            return

        if not available_ports:
            self._handle_startup_connect_failure("未找到可用设备")
            return

        selected_port = min(int(port) for port in available_ports)
        self._set_device_status(
            f"设备状态：已发现设备，正在连接 USB端口 {selected_port}...",
            "warning"
        )

        
        
        self._set_device_status(
            f"设备状态：正在初始化 USB端口 {selected_port}（约 5 秒）...",
            "warning"
        )
        self._sensor_init_worker = SensorInitWorker(selected_port, request_id)
        self._sensor_init_worker.initialized.connect(
            self._on_startup_sensor_ready, QtCore.Qt.QueuedConnection)
        self._sensor_init_worker.failed.connect(
            self._on_startup_sensor_failed, QtCore.Qt.QueuedConnection)
        self._sensor_init_worker.finished.connect(self._sensor_init_worker.deleteLater)
        self._sensor_init_worker.start()

    @QtCore.pyqtSlot(int, object)
    def _on_startup_sensor_ready(self, request_id, sensor_obj):
        self._sensor_init_worker = None
        if request_id != self._startup_connect_request_id:
            
            
            try:
                sensor_obj.release()
            except Exception:
                pass
            self._resume_pending_manual_connect()
            return

        try:
            self._apply_connected_sensor(sensor_obj, sensor_obj.usb_id)

            print(f"启动自动连接USB端口: {sensor_obj.usb_id}")
        except Exception as e:
            self.stop_sensor_worker()
            try:
                sensor_obj.release()
            except Exception:
                pass
            self._handle_startup_connect_failure(str(e))

    @QtCore.pyqtSlot(int, str)
    def _on_startup_sensor_failed(self, request_id, error_msg):
        self._sensor_init_worker = None
        if request_id != self._startup_connect_request_id:
            self._resume_pending_manual_connect()
            return
        self._handle_startup_connect_failure(error_msg)

    def _resume_pending_manual_connect(self):
        """孤儿传感器已释放后，续跑用户中途发起的“更换设备”流程。"""
        pending = self._manual_connect_pending_args
        self._manual_connect_pending_args = None
        if pending is not None:
            self._manual_connect_via_port_selector(*pending)

    def _on_startup_init_cancel_timeout(self):
        """兜底：后台构造 20s 仍未结束（SDK 调用挂死等极端情况），强制终止并继续。"""
        worker = getattr(self, '_sensor_init_worker', None)
        if worker is not None and worker.isRunning():
            print("传感器初始化线程超时（20s），强制终止")
            worker.terminate()
            worker.wait(1000)
            self._sensor_init_worker = None
        self._resume_pending_manual_connect()

    def _set_curve_x_ranges(self):
        for plot in (self.pgwin_tangent, self.pgwin_normal):
            plot.getViewBox().setXRange(float(self.time_axis[0]), 0.0, padding=0)

    def _apply_fixed_curve_ranges(self):
        self._set_curve_x_ranges()
        self.pgwin_tangent.getViewBox().setYRange(-2000.0, 2000.0, padding=0)
        self.pgwin_normal.getViewBox().setYRange(self.fz_fixed_y_min, self.fz_fixed_y_max, padding=0)

    def toggle_auto_calibration(self):
        if self.sensor is None:
            return
        self.sensor.auto_calibration_enabled = not self.sensor.auto_calibration_enabled
        self._sync_autocal_toggle()
        print(f"自动校准已{'开启' if self.sensor.auto_calibration_enabled else '关闭'}")

    def _sync_autocal_toggle(self):
        """将按钮文案与 sensor 的自动校准开关状态对齐。"""
        if self.sensor is None:
            self.autocal_toggle_btn.setText("自动校准：--")
            return
        status_text = "开启" if self.sensor.auto_calibration_enabled else "关闭"
        self.autocal_toggle_btn.setText(f"自动校准：{status_text}")

    def toggle_dynamic_range(self):
        self.dynamic_range_enabled = not self.dynamic_range_enabled
        status_text = "开启" if self.dynamic_range_enabled else "关闭"
        self.dynamic_range_btn.setText(f"Y轴自适应：{status_text}")

        if self.dynamic_range_enabled:
            self._set_curve_x_ranges()
            self.align_global_view()
        else:
            self._apply_fixed_curve_ranges()
        print(f"Y轴自适应已{status_text}")

    def align_global_view(self):
        tangent_all = np.concatenate([self.fx_hist, self.fy_hist])
        finite_t = tangent_all[np.isfinite(tangent_all)]
        if len(finite_t) > 0:
            t_min, t_max = np.min(finite_t), np.max(finite_t)
            view_min_t = min(float(t_min), -2000.0)
            view_max_t = max(float(t_max), 2000.0)
            margin_t = (view_max_t - view_min_t) * 0.1
            self.pgwin_tangent.getViewBox().setYRange(view_min_t - margin_t, view_max_t + margin_t, padding=0)

        finite_n = self.fz_hist[np.isfinite(self.fz_hist)]
        if len(finite_n) > 0:
            n_max = max(float(np.max(finite_n)), 0.0)
            margin_n = max(n_max * 0.1, 50.0)
            view_max_n = max(self.fz_auto_y_min + self.fz_auto_min_span, n_max + margin_n)
            self.pgwin_normal.getViewBox().setYRange(self.fz_auto_y_min, view_max_n, padding=0)

    def _on_heatmap_scale_changed(self, value):
        self.heatmap_scale = value / 10.0
        self.heatmap_scale_label.setText(f"{self.heatmap_scale:.0f}")

    def _on_scale_minus(self):
        new_val = max(self.heatmap_scale_slider.minimum(),
                      self.heatmap_scale_slider.value() - 10)
        self.heatmap_scale_slider.setValue(new_val)

    def _on_scale_plus(self):
        new_val = min(self.heatmap_scale_slider.maximum(),
                      self.heatmap_scale_slider.value() + 10)
        self.heatmap_scale_slider.setValue(new_val)

    def _on_arrow_scale_changed(self, value):
        self.arrow_xy_scale = value / 10.0
        self.arrow_scale_label.setText(f"{self.arrow_xy_scale:.0f}")

    def _on_arrow_scale_minus(self):
        new_val = max(self.arrow_scale_slider.minimum(),
                      self.arrow_scale_slider.value() - 10)
        self.arrow_scale_slider.setValue(new_val)

    def _on_arrow_scale_plus(self):
        new_val = min(self.arrow_scale_slider.maximum(),
                      self.arrow_scale_slider.value() + 10)
        self.arrow_scale_slider.setValue(new_val)

    def _on_arrow_z_scale_changed(self, value):
        self.arrow_z_scale = value / 10.0
        self.arrow_z_scale_label.setText(f"{self.arrow_z_scale:.0f}")

    def _on_arrow_z_scale_minus(self):
        new_val = max(self.arrow_z_scale_slider.minimum(),
                      self.arrow_z_scale_slider.value() - 10)
        self.arrow_z_scale_slider.setValue(new_val)

    def _on_arrow_z_scale_plus(self):
        new_val = min(self.arrow_z_scale_slider.maximum(),
                      self.arrow_z_scale_slider.value() + 10)
        self.arrow_z_scale_slider.setValue(new_val)

    def _apply_device_slider_config(self, device_type=None):
        """根据设备类型设置滑块范围和默认值，并重置到默认值。"""
        if device_type is None:
            device_type = self.device_type
        else:
            self.device_type = device_type

        config = DEVICE_SLIDER_CONFIG.get(device_type)
        if config is None:
            
            
            return

        
        hm = config['heatmap']
        self.heatmap_scale_slider.blockSignals(True)
        self.heatmap_scale_slider.setRange(hm['range'][0], hm['range'][1])
        self.heatmap_scale_slider.setValue(hm['default'])
        self.heatmap_scale_slider.blockSignals(False)
        self.heatmap_scale = hm['default'] / 10.0
        self.heatmap_scale_label.setText(f"{self.heatmap_scale:.0f}")

        
        axy = config['arrow_xy']
        self.arrow_scale_slider.blockSignals(True)
        self.arrow_scale_slider.setRange(axy['range'][0], axy['range'][1])
        self.arrow_scale_slider.setValue(axy['default'])
        self.arrow_scale_slider.blockSignals(False)
        self.arrow_xy_scale = axy['default'] / 10.0
        self.arrow_scale_label.setText(f"{self.arrow_xy_scale:.0f}")

        
        az = config['arrow_z']
        self.arrow_z_scale_slider.blockSignals(True)
        self.arrow_z_scale_slider.setRange(az['range'][0], az['range'][1])
        self.arrow_z_scale_slider.setValue(az['default'])
        self.arrow_z_scale_slider.blockSignals(False)
        self.arrow_z_scale = az['default'] / 10.0
        self.arrow_z_scale_label.setText(f"{self.arrow_z_scale:.0f}")

    def _on_reset_slider_defaults(self):
        """恢复当前设备类型的滑块默认值。"""
        self._apply_device_slider_config(self.device_type)

    def reset_view(self):
        self.planar_gl_widget.reset_camera_view()
        self.fingertip_gl_widget.reset_camera_view()
        if self.dynamic_range_enabled:
            self.toggle_dynamic_range()
        self._apply_fixed_curve_ranges()
        print("3D视图和力曲线范围已重置")

    def update_force_history(self, fx_sum, fy_sum, fz_sum):
        self.fx_smooth_value = self.xy_smooth_alpha * fx_sum + (1.0 - self.xy_smooth_alpha) * self.fx_smooth_value
        self.fy_smooth_value = self.xy_smooth_alpha * fy_sum + (1.0 - self.xy_smooth_alpha) * self.fy_smooth_value
        self.fx_hist[self.idx] = self.fx_smooth_value
        self.fy_hist[self.idx] = self.fy_smooth_value
        fz_val = max(0, fz_sum)
        self.fz_hist[self.idx] = fz_val

        idx = self.idx
        display_shift = -((idx + 1) % self.hist_len)
        self.fx_curve.setData(self.time_axis, np.roll(self.fx_hist, display_shift))
        self.fy_curve.setData(self.time_axis, np.roll(self.fy_hist, display_shift))
        self.fz_curve.setData(self.time_axis, np.roll(self.fz_hist, display_shift))

        self.idx = (self.idx + 1) % self.hist_len

        if self.dynamic_range_enabled:
            self.align_global_view()

    def calibrate(self):
        if self.sensor is None:
            return
        if self.button_processing:
            return
        self.button_processing = True
        if not self.stop_sensor_worker():
            self.button_processing = False
            return
        self._reset_radar_state()
        try:
            self.sensor.calibrate_baseline(10)
            print("校准完成！")
        except Exception as e:
            print(f"校准出错: {e}")
        finally:
            self._reset_rate_windows()
            if not self.should_exit:
                self.start_sensor_worker()
            QtCore.QTimer.singleShot(1000, lambda: setattr(self, 'button_processing', False))

    def quit_application(self):
        if getattr(self, '_shutdown_complete', False):
            return
        self.should_exit = True
        self._invalidate_startup_auto_connect()
        self._manual_connect_pending_args = None
        init_worker = getattr(self, '_sensor_init_worker', None)
        if init_worker is not None and init_worker.isRunning():
            init_worker.terminate()   
            init_worker.wait(1000)
        self.stop_sensor_worker()
        self.close()
        QtWidgets.QApplication.quit()

    def _manual_connect_via_port_selector(self, auto_scan, keep_main_on_cancel=False):
        self._invalidate_startup_auto_connect()
        if not self.stop_sensor_worker():
            return False
        self._reset_radar_state()

        
        
        
        init_worker = getattr(self, '_sensor_init_worker', None)
        if init_worker is not None and init_worker.isRunning():
            self._manual_connect_pending_args = (auto_scan, keep_main_on_cancel)
            self._set_device_status("设备状态：正在取消当前连接，请稍候...", "warning")
            QtCore.QTimer.singleShot(20000, self._on_startup_init_cancel_timeout)
            return False

        if self.sensor is not None:
            self.sensor.release()
            self.sensor = None
            self.current_port = None

        self._set_sensor_controls_enabled(False)
        self._sync_autocal_toggle()
        self.hide()

        port_selector = USBPortSelector(auto_scan=auto_scan)
        if port_selector.exec_() != QtWidgets.QDialog.Accepted:
            if keep_main_on_cancel:
                self.show()
                self._handle_startup_connect_failure("用户取消端口选择")
                return False
            self.quit_application()
            return False

        selected_port = port_selector.get_selected_port()
        if selected_port is None:
            if keep_main_on_cancel:
                self.show()
                self._handle_startup_connect_failure("未选择USB端口")
                return False
            self.quit_application()
            return False

        print(f"连接USB端口: {selected_port}")
        new_sensor = None
        try:
            new_sensor = TouchSensor(usb_id=selected_port)
            self._apply_connected_sensor(new_sensor, selected_port)
            self.show()
            return True
        except ValueError as e:
            if new_sensor is not None:
                self.stop_sensor_worker()
                new_sensor.release()
            QtWidgets.QMessageBox.critical(self, "设备标识错误", str(e) + "\n\n程序将强制退出。")
            self.quit_application()
            return False
        except Exception as e:
            if new_sensor is not None:
                self.stop_sensor_worker()
                new_sensor.release()
            print(f"初始化失败: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "初始化失败",
                f"无法初始化摄像头:\n{str(e)}\n\n请检查设备连接后重试。"
            )
            self.show()
            self.sensor = None
            self.current_port = None
            self._set_sensor_controls_enabled(False)
            self._reset_runtime_metrics("连接失败，请点击“更换设备”重新选择设备。")
            self._set_device_status(
                "设备状态：连接失败，请点击“更换设备”重新选择设备。",
                "error"
            )
            return False

    def back_to_port_selection(self):
        has_sensor = self.sensor is not None
        message = (
            "确定要更换设备吗？\n\n将断开当前连接并返回端口选择界面。"
            if has_sensor else
            "当前未连接设备，是否打开端口选择界面？"
        )
        confirm = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Question,
            "更换设备",
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            None
        )
        confirm.setDefaultButton(QtWidgets.QMessageBox.No)
        if confirm.exec_() != QtWidgets.QMessageBox.Yes:
            return
        self._manual_connect_via_port_selector(
            auto_scan=not has_sensor,
            keep_main_on_cancel=False,
        )
        return

    def closeEvent(self, event):
        if getattr(self, '_shutdown_complete', False):
            event.accept()
            return
        self.should_exit = True
        self._invalidate_startup_auto_connect()
        self._manual_connect_pending_args = None
        init_worker = getattr(self, '_sensor_init_worker', None)
        if init_worker is not None and init_worker.isRunning():
            init_worker.terminate()
            init_worker.wait(1000)
        self.stop_sensor_worker()
        
        
        if self.sensor is not None:
            self.sensor.release()
            self.sensor = None
        self._stop_radar_feature_worker()
        self._shutdown_complete = True
        super().closeEvent(event)

    def handle_key_press(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Q:
            self.quit_application()
        elif key == QtCore.Qt.Key_C:
            self.calibrate()
        else:
            super().keyPressEvent(event)

    @QtCore.pyqtSlot()
    def _consume_sensor_frame(self):
        worker = self._sensor_worker
        if self.should_exit or worker is None or self.sender() is not worker:
            return
        packet = worker.take_latest_frame()
        if packet is not None:
            self._on_new_frame(packet)

    @QtCore.pyqtSlot(dict)
    def _on_new_frame(self, packet):
        """Render the latest completed sensor snapshot on the GUI thread."""
        if self.should_exit:
            return

        try:
            
            fx_t = packet['fx_t']
            fy_t = packet['fy_t']
            fz_t = packet['fz_t']
            fz_raw = packet.get('fz_raw', float(fz_t))
            fx_raw = packet.get('fx_raw', float(fx_t))
            fy_raw = packet.get('fy_raw', float(fy_t))

            
            fz_calib = packet['fz_matrix']
            fx_calib = packet['fx_matrix']
            fy_calib = packet['fy_matrix']
            
            

            
            target_index = 1 if self.device_type == 'curved' else 0
            if self._current_pressure_index != target_index:
                self.pressure_stack.setCurrentIndex(target_index)
                self._current_pressure_index = target_index

            if self.device_type == 'curved':
                self.update_fingertip_view(fz_calib, fx_calib, fy_calib)
            else:
                
                self.update_planar_board_view(fz_calib, fx_calib, fy_calib)
            
            if (
                packet['hue_matrix'] is not None
                and packet['flow_matrix'] is not None
            ):
                self._radar_feature_worker.submit({
                    "frame_id": packet['seq'],
                    "generation": self._radar_generation,
                    "timestamp": packet['timestamp'],
                    "contact_threshold": packet['contact_threshold'],
                    "fz": fz_t,
                    "hue_matrix": packet['hue_matrix'],
                    "flow_matrix": packet['flow_matrix'],
                    "fx_matrix": fx_calib,
                    "fy_matrix": fy_calib,
                    "fz_matrix": fz_calib,
                })

            
            if fz_calib is None:
                return
            h, w = fz_calib.shape[:2]
            preview_gray = np.clip(fz_calib * self.heatmap_scale, 0, 255).astype(np.uint8)
            

            preview_bgr = cv2.applyColorMap(preview_gray, cv2.COLORMAP_JET)
            
            preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            ph, pw = preview_rgb.shape[:2]
            qimg = QImage(preview_rgb.data, pw, ph, 3 * pw, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(
                self.side_map_label.contentsRect().size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.side_map_label.setPixmap(pix)
            
            self.fx_value_label.setText(f"Fx: {fx_t} mN")
            self.fy_value_label.setText(f"Fy: {fy_t} mN")
            self.fz_value_label.setText(f"Fz: {fz_t} mN")

            self.update_force_history(fx_t, fy_t, fz_t)

        except Exception as e:
            print(f"数据处理出错: {e}")

    @QtCore.pyqtSlot(str)
    def _on_worker_error(self, error_msg):
        """Slot: handles sensor worker errors (emitted from QThread)."""
        worker = self._sensor_worker
        if self.should_exit or worker is None or self.sender() is not worker:
            return
        print(f"传感器线程错误: {error_msg}")
        self.should_exit = True
        self.stop_sensor_worker()
        QtWidgets.QMessageBox.critical(
            self, "相机连接错误",
            f"{error_msg}\n\n程序将强制退出。"
        )
        self.quit_application()

    @QtCore.pyqtSlot()
    def _consume_radar_result(self):
        worker = self._radar_feature_worker
        if self.should_exit or worker is None or self.sender() is not worker:
            return
        result = worker.take_latest_result()
        if result is not None:
            self._on_radar_ready(result)

    @QtCore.pyqtSlot(object)
    def _on_radar_ready(self, result):
        """Slot: 接收雷达特征计算结果，更新中间面板雷达图与状态文字。"""
        if self.should_exit or result["generation"] != self._radar_generation:
            return

        features = result["features"]
        signed_values = {
            index: float(features.get(RadarComputeWorker.FEATURE_NAMES[index], 0.0))
            for index in RadarComputeWorker.SIGNED_FEATURE_INDICES
        }
        self.radar_plot.update_radar(result["xs"], result["ys"], signed_values)

        contact = "接触" if features.get("是否接触", 0.0) > 0.5 else "空载"
        slide = "滑移" if features.get("是否滑移", 0.0) > 0.5 else "-"
        self.radar_status_label.setText(f"状态: {contact} | {slide}")

    def _reset_radar_state(self):
        """校准/换设备后清空雷达跨帧状态与绘制。"""
        self._radar_generation += 1
        self._radar_feature_worker.request_reset()
        self.radar_plot.clear_radar()
        self.radar_status_label.setText("状态: 等待数据")
    def _stop_radar_feature_worker(self):
        worker = getattr(self, "_radar_feature_worker", None)
        self._radar_feature_worker = None
        if worker is None:
            return
        worker.close_mailbox()
        worker.requestInterruption()
        worker.wake()
        if not worker.wait(2000):
            print("雷达特征线程未能在超时前停止")
            worker.terminate()
            worker.wait()

    def _get_planar_arrow_cache(self, w, h, grid_nx=16, grid_ny=16):
        key = (int(w), int(h), int(grid_nx), int(grid_ny))
        if self._planar_arrow_cache_key == key and self._planar_arrow_cache is not None:
            return self._planar_arrow_cache

        gx_1d = np.linspace(0, w - 1, grid_nx).astype(np.int32)
        gy_1d = np.linspace(0, h - 1, grid_ny).astype(np.int32)
        gx, gy = np.meshgrid(gx_1d, gy_1d)
        gx = gx.reshape(-1).astype(np.int32)
        gy = gy.reshape(-1).astype(np.int32)
        half_win = max(1, int(round((w / grid_nx) / 2)))
        y0 = np.maximum(0, gy - half_win)
        y1 = np.minimum(h, gy + half_win + 1)
        x0 = np.maximum(0, gx - half_win)
        x1 = np.minimum(w, gx + half_win + 1)
        area = np.maximum((y1 - y0) * (x1 - x0), 1).astype(np.float32)

        self._planar_arrow_cache_key = key
        self._planar_arrow_cache = {
            'gx': gx,
            'gy': gy,
            'x0': x0,
            'x1': x1,
            'y0': y0,
            'y1': y1,
            'area': area,
        }
        return self._planar_arrow_cache

    @staticmethod
    def _window_means_from_cache(matrix, cache):
        integral = np.pad(
            matrix.astype(np.float32, copy=False).cumsum(axis=0).cumsum(axis=1),
            ((1, 0), (1, 0)),
            mode='constant'
        )
        sums = (
            integral[cache['y1'], cache['x1']]
            - integral[cache['y0'], cache['x1']]
            - integral[cache['y1'], cache['x0']]
            + integral[cache['y0'], cache['x0']]
        )
        return (sums / cache['area']).astype(np.float32)


    def update_planar_board_view(self, fz_calib, fx_calib, fy_calib):
        """使用标定后的矩阵渲染平面视图：热力图用 fz_calib，箭头用 fx_calib/fy_calib"""
        if fz_calib is None or fx_calib is None or fy_calib is None:
            return

        
        fz_calib = cv2.flip(fz_calib, 0)
        fx_calib = cv2.flip(fx_calib, 0)
        fy_calib = cv2.flip(fy_calib, 0)

        
        gray_img = np.clip(fz_calib * self.heatmap_scale, 0, 255).astype(np.uint8)
        heatmap = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        h, w = fz_calib.shape[:2]

        
        grid_nx, grid_ny = 16, 16
        gx_1d = np.linspace(0, w - 1, grid_nx).astype(int)
        gy_1d = np.linspace(0, h - 1, grid_ny).astype(int)
        gx, gy = np.meshgrid(gx_1d, gy_1d)
        gx, gy = gx.reshape(-1), gy.reshape(-1)

        color_gain = self.arrow_color_gain  
        visual_key = (
            float(self.arrow_xy_scale),
            float(self.arrow_z_scale),
            float(self.arrow_color_gain),
            float(self.arrow_min_force),
            float(self.arrow_min_len),
        )
        if self._planar_arrow_visual_key != visual_key:
            self.cached_arrow_segments_2d = []
            self.cached_arrow_colors_rgba = []
            self.arrow_update_counter = 0
            self._planar_arrow_visual_key = visual_key

        need_update_arrows = (
                not self.cached_arrow_segments_2d or
                (self.arrow_update_counter % self.arrow_update_interval == 0)
        )

        if need_update_arrows:
            cache = self._get_planar_arrow_cache(w, h, grid_nx, grid_ny)
            fx_avg = self._window_means_from_cache(fx_calib, cache)
            fy_avg = self._window_means_from_cache(fy_calib, cache)
            fz_avg = self._window_means_from_cache(fz_calib, cache)

            gx_f = cache['gx'].astype(np.float32)
            gy_f = cache['gy'].astype(np.float32)
            mag_xy = np.hypot(fx_avg, fy_avg)
            mag_3d = np.hypot(mag_xy, fz_avg)
            active = mag_3d >= self.arrow_min_force

            ex = gx_f.copy()
            ey = gy_f.copy()
            dz = np.full_like(gx_f, float(self.arrow_min_len), dtype=np.float32)
            if np.any(active):
                ex[active] = np.clip(gx_f[active] + fx_avg[active] * self.arrow_xy_scale, 0, w - 1)
                ey[active] = np.clip(gy_f[active] + fy_avg[active] * self.arrow_xy_scale, 0, h - 1)
                dz[active] = fz_avg[active] * self.arrow_z_scale

            arrow_matrix = np.column_stack([gx_f, gy_f, ex, ey, dz]).astype(np.float32)
            color_levels = np.clip(mag_3d * color_gain, 0, 230).astype(np.int32)
            colors = self.planar_gl_widget.jet_lut_rgba[color_levels].copy()
            colors[:, :3] = np.maximum(colors[:, :3], 0.35)
            colors[:, 3] = 1.0

            self.cached_arrow_segments_2d = [tuple(row) for row in arrow_matrix]
            self.cached_arrow_colors_rgba = [tuple(row) for row in np.repeat(colors, 2, axis=0)]

        self.arrow_update_counter += 1

        self.planar_gl_widget.set_board_image_bgr(heatmap)
        self.planar_gl_widget.set_arrow_vectors(self.cached_arrow_segments_2d, self.cached_arrow_colors_rgba)

    def update_fingertip_view(self, fz_calib, fx_calib, fy_calib):
        """更新指尖模式视图：在1/4球面上绘制力箭头"""
        if fz_calib is None or fx_calib is None or fy_calib is None:
            return

        h, w = fz_calib.shape[:2]
        planar_top_scale = (
                1.10
                * float(self.planar_gl_widget.display_scale)
                * float(self.planar_gl_widget.overlay_to_board_ratio)
        )
        source_span = max(float(w), float(h))
        planar_span = source_span * planar_top_scale
        fingertip_span = float(self.fingertip_gl_widget.fingertip_radius) * 2.0
        fingertip_xy_scale = fingertip_span / max(source_span, 1.0)
        fingertip_z_scale = fingertip_span / max(planar_span, 1.0)

        self.fingertip_gl_widget.set_planar_arrow_visual_params(
            self.arrow_xy_scale,
            self.arrow_z_scale,
            self.arrow_color_gain,
            self.arrow_min_force,
            self.arrow_min_len,
            xy_world_scale=fingertip_xy_scale,
            z_world_scale=fingertip_z_scale
        )
        self.fingertip_gl_widget.set_fingertip_data(fz_calib, fx_calib, fy_calib)


        
if __name__ == "__main__":
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    dpi_rounding_policy = getattr(QtCore.Qt, 'HighDpiScaleFactorRoundingPolicy', None)
    dpi_pass_through = getattr(dpi_rounding_policy, 'PassThrough', None)
    if (
            hasattr(QtWidgets.QApplication, 'setHighDpiScaleFactorRoundingPolicy') and
            dpi_pass_through is not None
    ):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            dpi_pass_through
        )

    app = QtWidgets.QApplication([])

    
    icon_path = _resource_path("logo/logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))
    else:
        print(f"警告：未找到窗口图标文件 {icon_path}，使用默认图标")

    surf_win = Force3DSurfaceApp(sensor=None, max_hist_time=3)
    surf_win.show()
    QtCore.QTimer.singleShot(0, surf_win.begin_startup_auto_connect)

    
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass

    print("程序已直接进入主界面，启动后会在后台自动扫描并连接设备。")

    app.exec_()
    app.quit()
    sys.exit(0)

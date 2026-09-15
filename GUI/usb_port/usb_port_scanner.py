




import cv2
from typing import List, Optional, Tuple
import time
import sys
import os


def _get_sonix_cv_indices() -> List[int]:
    """获取 Sonix 相机的 OpenCV 端口号列表（复用 API 层 VID/PID 匹配）。

    成功时返回正确的 OpenCV 索引列表；失败或 SDK 不可用时返回空列表。
    """
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from api.hardware.xu_camera import XuCamera
        return XuCamera.enumerate_cv_indices()
    except Exception as e:
        print(f"[Scanner] Sonix enumeration unavailable: {e}")
        return []


class PortInfo:
    """端口信息类"""
    
    def __init__(self, port: int, cap=None):
        self.port = port
        self.cap = cap
        self.name = f"USB端口{port}"
        self.resolution = (0, 0)
        self.fps = 0.0
        self.status = "未知"
        self.is_available = False
        self.last_test_time = None
        
        if cap and cap.isOpened():
            self._get_port_info()
    
    def _get_port_info(self):
        """获取端口详细信息"""
        try:
            if self.cap and self.cap.isOpened():
                
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.resolution = (width, height)
                
                
                
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.status = "已连接"
                    self.is_available = True
                else:
                    self.status = "连接失败"
                    self.is_available = False
            else:
                self.status = "无法打开"
                self.is_available = False
        except Exception as e:
            self.status = f"错误: {str(e)}"
            self.is_available = False
    
    def _measure_fps(self, window=0.5, max_frames=30):
        """连续读帧实测帧率。

        CAP_PROP_FPS 是驱动上报的名义值（Linux 恒 30.0、Windows 恒 0.0），不可信，
        这里改为按时间戳实测：预热首帧后，在 window 秒内连续读帧，用成功读到的
        帧数反推 fps。返回 (fps, success_count)。
        """
        if not self.cap or not self.cap.isOpened():
            return 0.0, 0

        
        self.cap.read()

        start = time.perf_counter()
        success_count = 0
        while success_count < max_frames and time.perf_counter() - start < window:
            ret, frame = self.cap.read()
            if not (ret and frame is not None and frame.size > 0):
                break
            success_count += 1

        elapsed = time.perf_counter() - start
        fps = success_count / elapsed if elapsed > 0 else 0.0
        return fps, success_count

    def _verify_stream(self, max_frames=5) -> int:
        """只验证视频流是否正常，不测帧率。返回成功读到的帧数。"""
        if not self.cap or not self.cap.isOpened():
            return 0
        
        self.cap.read()
        success_count = 0
        for _ in range(max_frames):
            ret, frame = self.cap.read()
            if not (ret and frame is not None and frame.size > 0):
                break
            success_count += 1
        return success_count

    def test_connection(self, measure_fps: bool = True) -> Tuple[bool, str]:
        """测试端口连接。

        measure_fps=True：扫描阶段用，顺带实测帧率存入 self.fps；
        measure_fps=False：端口选择器“测试连接”按钮用，只验证流、不重测帧率，
        沿用扫描时测得的值。
        """
        try:
            if not self.cap or not self.cap.isOpened():
                return False, "端口未打开"
            
            if measure_fps:
                
                fps, success_count = self._measure_fps()
                self.fps = fps
            else:
                success_count = self._verify_stream()
            
            if success_count >= 2:  
                self.status = "连接正常"
                self.is_available = True
                self.last_test_time = time.time()
                return True, "连接测试成功"
            else:
                self.status = "连接不稳定"
                self.is_available = False
                return False, "视频流不稳定"
                
        except Exception as e:
            self.status = f"测试错误: {str(e)}"
            self.is_available = False
            return False, f"测试失败: {str(e)}"
    
    def get_info_string(self) -> str:
        """获取端口信息字符串"""
        if self.is_available:
            return f"{self.name} - {self.resolution[0]}x{self.resolution[1]} - {self.fps:.0f}fps - {self.status}"
        else:
            return f"{self.name} - {self.status}"
    
    def release(self):
        """释放端口资源"""
        if self.cap:
            self.cap.release()
            self.cap = None

class USBPortScanner:
    """USB端口扫描类"""
    
    def __init__(self, scan_range: Tuple[int, int] = (0, 10)): 
        self.scan_range = scan_range
        self.available_ports: List[PortInfo] = []
    
    def scan_available_ports(self, progress_callback=None) -> List[PortInfo]:
        """扫描可用的 Sonix USB 摄像头端口。

        两层扫描：
        第一层：通过 Sonix SDK 或 sysfs 获取 Sonix 设备端口列表，不打开视频流
        第二层：仅对第一层筛选出的端口用 OpenCV 验证视频流。

        未找到 Sonix 设备时返回空列表。
        """
        print("开始扫描USB端口...")
        self.available_ports = []

        
        sonix_ports = _get_sonix_cv_indices()

        if sonix_ports:
            print(f"检测到 {len(sonix_ports)} 个 Sonix 设备端口: {sonix_ports}，仅验证这些端口")
            scan_targets = sonix_ports
        else:
            print("未检测到 Sonix 设备")
            scan_targets = []

        total = len(scan_targets)
        for idx, port in enumerate(scan_targets):
            current = idx + 1
            print(f"验证端口 {port} ({current}/{total})...")
            port_info = self._scan_single_port(port)
            if port_info:
                self.available_ports.append(port_info)
                print(f"端口 {port} 可用: {port_info.get_info_string()}")
            else:
                print(f"端口 {port} 不可用")
            
            if port_info:
                port_info.release()
            if progress_callback:
                progress_callback(current, total, port)

        print(f"扫描完成，找到 {len(self.available_ports)} 个可用端口")
        return self.available_ports
    
    @staticmethod
    def _open_capture(port: int):
        """根据平台打开摄像头。
        Linux → V4L2（避免 FFMPEG 探测耗时）
        macOS → AVFoundation
        Windows → DirectShow（与 COM 枚举端口号一致，且避免 MSMF 多相机兼容问题）"""
        if sys.platform.startswith('linux'):
            return cv2.VideoCapture(port, cv2.CAP_V4L2)
        elif sys.platform.startswith('darwin'):
            return cv2.VideoCapture(port, cv2.CAP_AVFOUNDATION)
        else:
            return cv2.VideoCapture(port, cv2.CAP_DSHOW)

    def _scan_single_port(self, port: int) -> Optional[PortInfo]:
        """扫描单个端口。

        Linux 优化：优先尝试 MJPG 格式。Sonix 摄像头在 Linux 通用 UVC
        驱动下以默认 YUYV 格式启动视频流时，OpenCV V4L2 后端的
        select() 会触发两次 10 秒超时（硬编码），导致单端口扫描耗时
        20+ 秒。MJPG 格式下摄像头流启动路径更短，可避免超时。
        Windows：使用默认后端（DirectShow），先尝试默认格式再尝试 MJPG。
        """
        is_linux = sys.platform.startswith('linux')
        try:
            cap = self._open_capture(port)
            if not cap.isOpened():
                cap.release()
                return None

            if is_linux:
                
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                port_info = PortInfo(port, cap)
                success, _ = port_info.test_connection()
                if success:
                    return port_info

                
                port_info.release()
                cap = cv2.VideoCapture(port, cv2.CAP_V4L2)
                if cap.isOpened():
                    port_info = PortInfo(port, cap)
                    success, _ = port_info.test_connection()
                    if success:
                        return port_info
                    port_info.release()
                return None
            else:
                
                port_info = PortInfo(port, cap)
                success, _ = port_info.test_connection()
                if success:
                    return port_info

                
                port_info.release()
                cap = self._open_capture(port)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    port_info = PortInfo(port, cap)
                    success, _ = port_info.test_connection()
                    if success:
                        return port_info
                    port_info.release()
                return None

        except Exception as e:
            print(f"扫描端口 {port} 时出错: {e}")
            return None
    
    def get_port_by_number(self, port: int) -> Optional[PortInfo]:
        """根据端口号获取端口信息"""
        for port_info in self.available_ports:
            if port_info.port == port:
                return port_info
        return None
    
    def release_all_ports(self):
        """释放所有端口资源"""
        for port_info in self.available_ports:
            port_info.release()
        self.available_ports = []

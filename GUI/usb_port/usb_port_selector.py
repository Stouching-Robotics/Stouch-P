



import sys
import cv2
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from .usb_port_scanner import USBPortScanner, PortInfo
import json
from datetime import datetime

class USBPortSelector(QtWidgets.QDialog):
    """USB端口选择对话框"""
    
    def __init__(self, parent=None, auto_scan=True):
        super().__init__(parent)
        self.scanner = USBPortScanner()
        self.current_port_info = None
        self._connection_tested = False  
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self._watchdog_timer = QTimer()
        self._watchdog_timer.timeout.connect(self._check_camera_alive)
        self._watchdog_timer.setInterval(500)  
        
        self.config_file = "./usb_port.json"

        
        self.setWindowTitle("USB端口选择")
        self.setModal(True)
        self.resize(800, 600)

        
        self.setup_ui()

        
        if auto_scan:
            self._set_status("正在扫描USB端口，请稍候...", "warning")
            
            QTimer.singleShot(0, self.scan_ports)
        else:
            self._set_status("未扫描", "error")
    
    def setup_ui(self):
        """设置用户界面"""
        layout = QtWidgets.QVBoxLayout()
        
        
        title_label = QtWidgets.QLabel("准备连接设备")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        
        main_layout = QtWidgets.QHBoxLayout()
        
        
        left_layout = QtWidgets.QVBoxLayout()
        
        
        port_layout = QtWidgets.QHBoxLayout()
        port_layout.addWidget(QtWidgets.QLabel("选择端口:"))
        
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setMinimumWidth(250)
        self.port_combo.currentTextChanged.connect(self.on_port_selected)
        port_layout.addWidget(self.port_combo)
        
        
        self.rescan_btn = QtWidgets.QPushButton("重新扫描")
        self.rescan_btn.clicked.connect(self.scan_ports)
        self.rescan_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: bold; }")
        port_layout.addWidget(self.rescan_btn)
        
        left_layout.addLayout(port_layout)

        
        self.status_label = QtWidgets.QLabel("状态: 未扫描")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(36)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #f39c12;
                font-weight: bold;
                font-size: 15px;
                background-color: #fff8e1;
                border: 2px solid #f39c12;
                border-radius: 8px;
                padding: 8px 12px;
            }
        """)
        left_layout.addWidget(self.status_label)

        
        self.port_info_label = QtWidgets.QLabel("请选择端口查看信息")
        self.port_info_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-family: monospace;
            }
        """)
        self.port_info_label.setWordWrap(True)
        self.port_info_label.setMinimumHeight(120)
        self.port_info_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
        left_layout.addWidget(self.port_info_label)
        
        
        test_disconnect_layout = QtWidgets.QHBoxLayout()
        self.test_btn = QtWidgets.QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        self.test_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; }")
        self.test_btn.setEnabled(False)
        test_disconnect_layout.addWidget(self.test_btn)

        self.disconnect_btn = QtWidgets.QPushButton("断开连接")
        self.disconnect_btn.clicked.connect(self.disconnect_port)
        self.disconnect_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }")
        self.disconnect_btn.setEnabled(False)
        test_disconnect_layout.addWidget(self.disconnect_btn)

        left_layout.addLayout(test_disconnect_layout)

        left_layout.addStretch()
        main_layout.addLayout(left_layout, 1)

        
        right_layout = QtWidgets.QVBoxLayout()

        
        preview_title = QtWidgets.QLabel("摄像头预览")
        preview_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        preview_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(preview_title)

        
        self.preview_label = QtWidgets.QLabel("连接端口后自动预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(400, 300)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 10px;
                font-size: 14px;
            }
        """)
        preview_container = QtWidgets.QHBoxLayout()
        preview_container.addStretch()
        preview_container.addWidget(self.preview_label)
        preview_container.addStretch()
        right_layout.addLayout(preview_container)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, 2)
        
        layout.addLayout(main_layout)
        
        
        button_layout = QtWidgets.QHBoxLayout()
        
        button_layout.addStretch()

        self.cancel_btn = QtWidgets.QPushButton("退出")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #95a5a6; color: white; font-weight: bold; }")
        button_layout.addWidget(self.cancel_btn)

        self.ok_btn = QtWidgets.QPushButton("进入主界面")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; }")
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def scan_ports(self):
        """扫描可用端口（仅检测 Sonix 设备）"""
        
        try:
            import pyi_splash
            pyi_splash.close()
        except (ImportError, ModuleNotFoundError):
            pass

        self.rescan_btn.setEnabled(False)
        self.rescan_btn.setText("扫描中...")
        self._set_status("正在检测设备，请稍候...", "warning")

        
        self.port_combo.clear()
        self.scanner.release_all_ports()

        
        QtWidgets.QApplication.processEvents()

        
        def on_progress(current, total, port):
            self._set_status(f"正在验证端口 {current}/{total}  (端口 {port})", "warning")
            QtWidgets.QApplication.processEvents()

        available_ports = self.scanner.scan_available_ports(progress_callback=on_progress)

        if available_ports:
            for port_info in available_ports:
                self.port_combo.addItem(port_info.get_info_string(), port_info.port)

            
            self.port_combo.setCurrentIndex(0)
            self.on_port_selected(self.port_combo.currentText())
            self._set_status("扫描完成，未连接", "error")
        else:
            self.port_combo.addItem("未找到可用端口")
            self.port_info_label.setText("未找到可用的USB摄像头端口")
            self._set_status("扫描完成，未找到设备", "error")

        self.rescan_btn.setEnabled(True)
        self.rescan_btn.setText("重新扫描")

    def _set_status(self, text, level="warning"):
        """统一设置状态标签文本与样式。

        level:
            'warning' — 黄色（扫描中）
            'success' — 绿色（已连接）
            'error'   — 红色（未连接 / 失败）
        """
        styles = {
            "warning": ("#f39c12", "#fff8e1", "#f39c12"),
            "success": ("#27ae60", "#e8f5e9", "#27ae60"),
            "error":   ("#e74c3c", "#fdecea", "#e74c3c"),
        }
        color, bg, border = styles.get(level, styles["warning"])
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                font-size: 15px;
                background-color: {bg};
                border: 2px solid {border};
                border-radius: 8px;
                padding: 8px 12px;
            }}
        """)
    
    def on_port_selected(self, text):
        """端口选择处理"""
        if not text or "未找到" in text:
            return
        
        
        port = self.port_combo.currentData()
        if port is None:
            return
        
        
        self.current_port_info = self.scanner.get_port_by_number(port)
        if self.current_port_info:
            self.update_port_info()
            self.test_btn.setEnabled(True)
            self._connection_tested = False  
        else:
            self.port_info_label.setText("端口信息获取失败")
            self.test_btn.setEnabled(False)
    
    def update_port_info(self):
        """更新端口信息显示"""
        if not self.current_port_info:
            return
        
        info_text = f"""
端口信息:
• 端口号: {self.current_port_info.port}
• 名称: {self.current_port_info.name}
• 分辨率: {self.current_port_info.resolution[0]}x{self.current_port_info.resolution[1]}
• 帧率: {self.current_port_info.fps:.2f} fps
• 状态: {self.current_port_info.status}
• 可用性: {'是' if self.current_port_info.is_available else '否'}
        """.strip()
        
        self.port_info_label.setText(info_text)
    
    def test_connection(self):
        """测试端口连接"""
        if not self.current_port_info:
            return

        self.test_btn.setEnabled(False)
        self.test_btn.setText("测试中...")

        
        if not self.current_port_info.cap or not self.current_port_info.cap.isOpened():
            self.current_port_info.cap = self._open_capture(self.current_port_info.port)

        success, message = self.current_port_info.test_connection(measure_fps=False)
        
        if success:
            self._set_status("已连接（如要切换相机请先断开连接）", "success")
            self._connection_tested = True
            self.port_combo.setEnabled(False)
            self.rescan_btn.setEnabled(False)
            self.test_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_preview()  
            self._watchdog_timer.start()  
        else:
            self._set_status(f"连接失败 - {message}", "error")
            self._connection_tested = False
            
            if self.current_port_info:
                self.current_port_info.release()
        
        self.test_btn.setEnabled(True)
        self.test_btn.setText("测试连接")
        self.update_port_info()

    def disconnect_port(self):
        """断开当前端口连接，恢复端口选择"""
        self.stop_preview()
        self._watchdog_timer.stop()
        self._connection_tested = False
        
        if self.current_port_info:
            self.current_port_info.release()
        self.port_combo.setEnabled(True)
        self.rescan_btn.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.ok_btn.setEnabled(True)
        self._set_status("未连接", "error")

    def _check_camera_alive(self):
        """看门狗：检测连接的相机是否中途断开"""
        if not self.current_port_info or not self.current_port_info.cap:
            self.disconnect_port()
            return
        ret, _ = self.current_port_info.cap.read()
        if not ret:
            self.disconnect_port()

    @staticmethod
    def _open_capture(port: int):
        """根据平台打开摄像头，与 TouchSensor.camera_init 保持一致。
        Linux 下强制 MJPG 格式，避免 V4L2 select() 10s 超时。"""
        if sys.platform.startswith('linux'):
            cap = cv2.VideoCapture(port, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            return cap
        elif sys.platform.startswith('darwin'):
            return cv2.VideoCapture(port, cv2.CAP_AVFOUNDATION)
        else:
            return cv2.VideoCapture(port)

    def start_preview(self):
        """开始预览"""
        if not self.current_port_info or not self.current_port_info.is_available:
            return

        try:
            if not self.current_port_info.cap or not self.current_port_info.cap.isOpened():
                self.current_port_info.cap = self._open_capture(self.current_port_info.port)

            if self.current_port_info.cap.isOpened():
                self.preview_timer.start(33)  
        except Exception:
            pass  

    def stop_preview(self):
        """停止预览"""
        self.preview_timer.stop()
        self.preview_label.setText("连接端口后自动预览")
        self._set_status("未连接", "error")
    
    def update_preview(self):
        """更新预览画面"""
        if not self.current_port_info or not self.current_port_info.cap:
            return
        
        try:
            ret, frame = self.current_port_info.cap.read()
            if ret and frame is not None:
                
                height, width = frame.shape[:2]
                max_width = 400
                max_height = 300
                
                if width > max_width or height > max_height:
                    scale = min(max_width / width, max_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                
                pixmap = QPixmap.fromImage(qt_image)
                self.preview_label.setPixmap(pixmap)
            else:
                self.stop_preview()
                QtWidgets.QMessageBox.warning(self, "错误", "无法获取视频流")
        except Exception as e:
            self.stop_preview()
            QtWidgets.QMessageBox.critical(self, "错误", f"预览更新时出错: {str(e)}")
    
    def save_config(self, selected_port):
        """保存配置"""
        try:
            config = {
                'last_selected_port': selected_port,
                'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'port_info': {
                    'port': selected_port,
                    'name': f'USB端口{selected_port}'
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"配置已保存: USB端口{selected_port}")
        except Exception as e:
            print(f"保存配置时出错: {e}")
    
    def get_selected_port(self):
        """获取选中的端口号"""
        if self.current_port_info:
            return self.current_port_info.port
        return None
    
    def accept(self):
        """确定按钮处理"""
        if not self.current_port_info:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择一个端口")
            return
        if not self._connection_tested:
            QtWidgets.QMessageBox.warning(self, "提示", "请先点击 [测试连接] 确认相机可用后再进入")
            return
        
        
        self.stop_preview()
        self._watchdog_timer.stop()
        self.current_port_info.cap.release()

        
        selected_port = self.get_selected_port()
        if selected_port is not None:
            self.save_config(selected_port)

        
        for port_info in self.scanner.available_ports:
            if port_info.port != selected_port:
                port_info.release()

        super().accept()

    def reject(self):
        """退出按钮处理"""
        self.stop_preview()
        self._watchdog_timer.stop()

        
        self.scanner.release_all_ports()

        super().reject()

    def closeEvent(self, event):
        """关闭事件处理"""
        self.stop_preview()
        self._watchdog_timer.stop()

        
        self.scanner.release_all_ports()

        event.accept()

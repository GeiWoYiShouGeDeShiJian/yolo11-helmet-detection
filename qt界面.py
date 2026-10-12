import os
import sys
import locale
import cv2
import numpy as np
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO

# ============ 修复Windows中文路径编码问题 ============
# 强制使用UTF-8编码
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None

# 设置本地化编码
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置Qt插件路径（使用原始字符串）
qt_plugin_path = r"C:\Users\筱筱\AppData\Local\Programs\Python\Python38\lib\site-packages\PyQt5\Qt5\plugins"

# 验证路径是否存在
if os.path.exists(qt_plugin_path):
    # 将路径转换为绝对路径并确保格式正确
    qt_plugin_path = os.path.abspath(qt_plugin_path)
    print(f"[QT修复] 插件路径: {qt_plugin_path}")
    print(f"[QT修复] 路径存在: {os.path.exists(qt_plugin_path)}")

    # 设置环境变量
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path

    # 打印检查（使用ASCII字符避免编码问题）
    print("[QT修复] 已设置QT_QPA_PLATFORM_PLUGIN_PATH环境变量")
else:
    print(f"[QT修复] 错误: 插件路径不存在: {qt_plugin_path}")

# 启用Qt调试信息（会显示详细的加载过程）
os.environ['QT_DEBUG_PLUGINS'] = '1'
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=true'


# ============ 自定义绘制函数 - 支持中文 ============
def draw_custom_boxes(image, results, conf_threshold=0.25):
    """
    自定义绘制检测框：修正类别、中文字体、不同颜色
    使用PIL绘制中文，然后转换回OpenCV格式
    
    Args:
        image: OpenCV格式的图像
        results: YOLO检测结果
        conf_threshold: 置信度阈值
    
    Returns:
        tuple: (标注后的图像, 佩戴安全帽数量, 未佩戴安全帽数量)
    """
    if not results:
        return image, 0, 0

    result = results[0]
    helmet_count = 0
    no_helmet_count = 0

    # 检查是否有检测结果
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.cpu().numpy()

        # 将OpenCV图像转换为PIL图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 加载中文字体
        font = _load_chinese_font()

        for box in boxes:
            # 获取框信息
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])

            # 跳过低置信度的检测
            if conf < conf_threshold:
                continue

            # 类别和颜色映射
            if cls == 0:
                label_text = "安全帽"
                color = (0, 0, 255)  # 蓝色 (RGB格式)
                helmet_count += 1
            else:  # cls == 1
                label_text = "未戴安全帽"
                color = (255, 0, 0)  # 红色 (RGB格式)
                no_helmet_count += 1

            # 添加置信度到标签
            label_text = f"{label_text} {conf:.2f}"

            # 绘制矩形框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 获取文本大小并绘制背景和文本
            text_width, text_height = _get_text_size(draw, label_text, font)
            draw.rectangle(
                [x1, y1 - text_height - 5, x1 + text_width, y1],
                fill=color
            )
            draw.text(
                (x1, y1 - text_height - 5),
                label_text,
                font=font,
                fill=(255, 255, 255)  # 白色
            )

        # 将PIL图像转换回OpenCV格式
        annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        annotated_image = image

    return annotated_image, helmet_count, no_helmet_count


def _load_chinese_font():
    """
    加载中文字体
    
    Returns:
        ImageFont: 加载的字体对象
    """
    # Windows系统字体路径
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "simhei.ttf",  # 当前目录下的字体
        "msyh.ttc"  # 当前目录下的字体
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, 20)
            except Exception:
                continue

    # 如果找不到中文字体，使用默认字体（英文）
    print("警告: 未找到中文字体，使用默认字体")
    return ImageFont.load_default()


def _get_text_size(draw, text, font):
    """
    获取文本大小
    
    Args:
        draw: ImageDraw对象
        text: 文本内容
        font: 字体对象
    
    Returns:
        tuple: (文本宽度, 文本高度)
    """
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        return text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except Exception:
        # 如果获取文本大小失败，使用估计值
        return len(text) * 10, 20


# ============ 视频处理线程类 ============
class VideoThread(QThread):
    """视频处理线程 - 修复掉帧和进度条问题"""
    frame_processed = pyqtSignal(np.ndarray, dict)
    progress_updated = pyqtSignal(int, int, float)
    video_info_ready = pyqtSignal(int, float)  # 发送视频信息
    playback_state_changed = pyqtSignal(bool)  # 播放状态改变信号

    def __init__(self, model, video_path, conf_threshold):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self._running = True
        self._paused = False  # 暂停状态
        self._seek_to_frame = -1
        self._current_frame_pos = 0
        self._lock = QMutex()
        self._pause_cond = QWaitCondition()  # 暂停条件
        self.total_frames = 0
        self.fps = 30

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {self.video_path}")
            return

        # 获取视频信息
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # 默认值

        # 发送视频信息到主线程
        self.video_info_ready.emit(self.total_frames, self.fps)

        frame_interval = 1000 / self.fps  # 每帧间隔（毫秒）

        while self._running and cap.isOpened():
            # 检查暂停状态
            self._lock.lock()
            if self._paused:
                self._pause_cond.wait(self._lock)
            self._lock.unlock()

            # 检查是否需要跳转到指定帧
            self._lock.lock()
            if self._seek_to_frame >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to_frame)
                self._current_frame_pos = self._seek_to_frame
                self._seek_to_frame = -1
            self._lock.unlock()

            ret, frame = cap.read()
            if not ret:
                # 发送最后一帧的进度
                self.progress_updated.emit(self.total_frames, self.total_frames, self.fps)
                break

            self._current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            start_time = time.time()

            # 进行检测（使用模型推理）
            try:
                results = self.model(frame, conf=self.conf_threshold, verbose=False)

                # 使用自定义绘制函数
                annotated_frame, helmet_count, no_helmet_count = draw_custom_boxes(
                    frame, results, self.conf_threshold
                )

                # 计算实际处理时间和FPS
                processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
                current_fps = 1000 / processing_time if processing_time > 0 else 0
                
                # 准备统计数据
                stats = {
                    'fps': current_fps,
                    'total': helmet_count + no_helmet_count,
                    'helmet': helmet_count,
                    'no_helmet': no_helmet_count
                }

                # 发送处理后的帧和进度
                self.frame_processed.emit(annotated_frame, stats)
                self.progress_updated.emit(self._current_frame_pos, self.total_frames, self.fps)

                # 控制播放速度，确保流畅
                sleep_time = max(1, int(frame_interval - processing_time))
                if sleep_time > 0:
                    self.msleep(sleep_time)

            except Exception as e:
                print(f"视频处理错误: {e}")
                # 发送原始帧和空统计
                self.frame_processed.emit(frame, {'fps': 0, 'total': 0, 'helmet': 0, 'no_helmet': 0})
                self.progress_updated.emit(self._current_frame_pos, self.total_frames, self.fps)
                self.msleep(int(frame_interval))

        cap.release()

    def seek_to(self, frame_num):
        """跳转到指定帧"""
        self._lock.lock()
        self._seek_to_frame = min(max(0, frame_num), self.total_frames - 1)
        self._lock.unlock()

    def toggle_pause(self):
        """切换暂停状态"""
        self._lock.lock()
        self._paused = not self._paused
        if not self._paused:
            self._pause_cond.wakeAll()
        self._lock.unlock()
        self.playback_state_changed.emit(self._paused)

    def stop(self):
        """停止线程"""
        self._running = False
        self._lock.lock()
        self._paused = False
        self._pause_cond.wakeAll()
        self._lock.unlock()
        self.wait(500)


# ============ 主窗口类 ============
class HelmetDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 模型和视频相关变量
        self.model = None
        self.cap = None
        self.video_writer = None
        
        # 状态标志
        self.is_camera_running = False
        self.is_recording = False
        self.is_video_playing = False
        self.is_seeking = False
        self.is_video_paused = False
        
        # 视频相关参数
        self.video_path = ""
        self.video_frames_total = 0
        self.video_current_frame = 0
        self.video_fps = 30
        
        # 摄像头相关参数
        self.camera_fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 其他变量
        self.current_frame = None
        self.video_thread = None

        self.init_ui()
        self.load_model()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("安全帽佩戴检测系统 - YOLOv11")
        self.setGeometry(100, 100, 1300, 850)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 顶部控制面板
        layout.addWidget(self._create_control_panel())

        # 视频进度条（初始隐藏）
        self._create_progress_panel()
        layout.addWidget(self.progress_panel)

        # 主显示区域
        layout.addWidget(self._create_display_panel())

        # 状态栏
        self.statusBar().showMessage("安全帽检测系统已就绪 | 中文标签已启用 | mAP50: 95.26%")

    def _create_control_panel(self):
        """创建顶部控制面板"""
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # 模型信息标签
        self.model_label = QLabel("模型状态: 加载中...")
        self.model_label.setStyleSheet("font-size: 14px; color: #333; padding: 5px;")
        control_layout.addWidget(self.model_label)

        control_layout.addStretch()

        # 置信度阈值滑块
        self.conf_label = QLabel("置信度: 0.25")
        self.conf_label.setMinimumWidth(80)
        control_layout.addWidget(self.conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(25)
        self.conf_slider.setMaximumWidth(150)
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        control_layout.addWidget(self.conf_slider)

        # 保存按钮
        self.save_btn = QPushButton("💾 保存当前")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_current)
        self.save_btn.setToolTip("保存当前检测结果")
        control_layout.addWidget(self.save_btn)

        # 录制按钮
        self.record_btn = QPushButton("⏺️ 开始录制")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setToolTip("录制检测视频")
        control_layout.addWidget(self.record_btn)

        return control_panel

    def _create_progress_panel(self):
        """创建视频进度条面板"""
        self.progress_panel = QWidget()
        self.progress_layout = QHBoxLayout(self.progress_panel)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(100)
        self.progress_layout.addWidget(self.time_label)

        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.setRange(0, 1000)  # 固定范围，便于拖动
        self.video_slider.setSingleStep(10)  # 设置步长，便于键盘控制
        self.video_slider.setPageStep(50)  # 设置页面步长
        self.video_slider.sliderPressed.connect(self.on_slider_pressed)
        self.video_slider.sliderReleased.connect(self.on_slider_released)
        self.video_slider.valueChanged.connect(self.on_slider_moved)
        self.progress_layout.addWidget(self.video_slider)

        # 播放/暂停按钮
        self.play_pause_btn = QPushButton("⏸️")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setMaximumWidth(40)
        self.play_pause_btn.setToolTip("播放/暂停")
        self.progress_layout.addWidget(self.play_pause_btn)

        self.progress_panel.hide()

    def _create_display_panel(self):
        """创建主显示区域"""
        display_panel = QWidget()
        display_layout = QHBoxLayout(display_panel)

        # 左侧：视频显示
        display_layout.addWidget(self._create_video_container())

        # 右侧：信息面板
        display_layout.addWidget(self._create_info_panel())

        return display_panel

    def _create_video_container(self):
        """创建视频显示容器"""
        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)

        # 图例说明
        legend_panel = QWidget()
        legend_layout = QHBoxLayout(legend_panel)
        legend_layout.setContentsMargins(0, 0, 0, 0)

        blue_legend = QLabel("🔵 蓝色框: 佩戴安全帽")
        blue_legend.setStyleSheet("color: blue; font-weight: bold; padding: 2px;")
        red_legend = QLabel("🔴 红色框: 未戴安全帽")
        red_legend.setStyleSheet("color: red; font-weight: bold; padding: 2px;")

        legend_layout.addWidget(blue_legend)
        legend_layout.addWidget(red_legend)
        legend_layout.addStretch()

        video_container_layout.addWidget(legend_panel)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(850, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            border: 2px solid #ccc; 
            background-color: #f0f0f0;
            border-radius: 5px;
        """)
        video_container_layout.addWidget(self.video_label)

        # 状态信息
        self.video_info_label = QLabel("请选择检测模式")
        self.video_info_label.setAlignment(Qt.AlignCenter)
        self.video_info_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        video_container_layout.addWidget(self.video_info_label)

        return video_container

    def _create_info_panel(self):
        """创建右侧信息面板"""
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)

        # 检测统计
        info_layout.addWidget(self._create_stats_group())

        # 控制按钮
        info_layout.addWidget(self._create_btn_group())

        # 历史记录
        info_layout.addWidget(self._create_history_group())

        # 添加说明
        info_layout.addWidget(self._create_help_group())

        info_layout.addStretch()
        return info_panel

    def _create_stats_group(self):
        """创建检测统计组"""
        stats_group = QGroupBox("📊 检测统计")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        stats_layout = QGridLayout()

        self.fps_label = QLabel("FPS: --")
        self.objects_label = QLabel("检测对象: 0")
        self.helmet_label = QLabel("🔵 佩戴安全帽: 0")
        self.no_helmet_label = QLabel("🔴 未戴安全帽: 0")
        self.save_status_label = QLabel("保存状态: 无")

        self.helmet_label.setStyleSheet("color: blue;")
        self.no_helmet_label.setStyleSheet("color: red;")

        stats_layout.addWidget(QLabel("性能:"), 0, 0)
        stats_layout.addWidget(self.fps_label, 0, 1)
        stats_layout.addWidget(QLabel("数量:"), 1, 0)
        stats_layout.addWidget(self.objects_label, 1, 1)
        stats_layout.addWidget(QLabel(""), 2, 0)  # 空白行
        stats_layout.addWidget(self.helmet_label, 3, 0, 1, 2)
        stats_layout.addWidget(self.no_helmet_label, 4, 0, 1, 2)
        stats_layout.addWidget(QLabel("保存:"), 5, 0)
        stats_layout.addWidget(self.save_status_label, 5, 1)

        stats_group.setLayout(stats_layout)
        return stats_group

    def _create_btn_group(self):
        """创建控制按钮组"""
        btn_group = QGroupBox("🎮 控制面板")
        btn_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        btn_layout = QVBoxLayout()

        self.camera_btn = QPushButton("📷 开启摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.camera_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 14px; }")

        self.image_btn = QPushButton("🖼️ 选择图片")
        self.image_btn.clicked.connect(self.select_image)
        self.image_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 14px; }")

        self.video_btn = QPushButton("🎬 选择视频")
        self.video_btn.clicked.connect(self.select_video)
        self.video_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 14px; }")

        self.stop_btn = QPushButton("⏹️ 停止检测")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-size: 14px; background-color: #ff6b6b; color: white; }")

        btn_layout.addWidget(self.camera_btn)
        btn_layout.addWidget(self.image_btn)
        btn_layout.addWidget(self.video_btn)
        btn_layout.addWidget(self.stop_btn)

        btn_group.setLayout(btn_layout)
        return btn_group

    def _create_history_group(self):
        """创建历史记录组"""
        history_group = QGroupBox("📁 最近保存")
        history_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        history_layout = QVBoxLayout()

        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        self.history_list.itemDoubleClicked.connect(self.open_saved_file)
        history_layout.addWidget(self.history_list)

        # 清空历史按钮
        clear_btn = QPushButton("清空记录")
        clear_btn.clicked.connect(self.clear_history)
        clear_btn.setMaximumWidth(100)
        history_layout.addWidget(clear_btn, alignment=Qt.AlignRight)

        history_group.setLayout(history_layout)
        return history_group

    def _create_help_group(self):
        """创建帮助信息组"""
        help_group = QGroupBox("ℹ️ 使用说明")
        help_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        help_layout = QVBoxLayout()

        help_text = QLabel(
            "🎯 功能说明：\n"
            "1. 修正类别识别\n"
            "2. 蓝色框 = 佩戴安全帽\n"
            "3. 红色框 = 未戴安全帽\n"
            "4. 完整中文标签显示\n\n"
            "操作说明：\n"
            "• 📷 开启摄像头：实时检测\n"
            "• 🖼️ 选择图片：检测并保存\n"
            "• 🎬 选择视频：检测视频（可拖动）\n"
            "• 💾 保存当前：保存检测结果\n"
            "• ⏺️ 录制：录制检测视频"
        )
        help_text.setStyleSheet("font-size: 12px; color: #666; line-height: 1.5;")
        help_layout.addWidget(help_text)

        help_group.setLayout(help_layout)
        return help_group

    def load_model(self):
        """加载YOLOv11模型"""
        try:
            model_path = 'runs/detect/helmet_final_phase/weights/best.pt'
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "警告", f"模型文件不存在: {model_path}")
                self.model_label.setText("模型状态: 未找到模型文件")
                return

            self.model = YOLO(model_path)
            self.model_label.setText("模型状态: ✅ 已加载 (YOLOv11s)")
            self.statusBar().showMessage("模型加载成功！中文标签已启用")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.model_label.setText("模型状态: ❌ 加载失败")

    def update_conf_threshold(self, value):
        """更新置信度阈值"""
        conf = value / 100.0
        self.conf_label.setText(f"置信度: {conf:.2f}")
        self.statusBar().showMessage(f"置信度阈值已更新: {conf:.2f}")

    def toggle_play_pause(self):
        """切换播放/暂停状态"""
        if self.is_video_playing and self.video_thread:
            self.video_thread.toggle_pause()

    def on_playback_state_changed(self, paused):
        """处理播放状态改变"""
        self.is_video_paused = paused
        if paused:
            self.play_pause_btn.setText("▶️")
            self.statusBar().showMessage("视频已暂停")
        else:
            self.play_pause_btn.setText("⏸️")
            self.statusBar().showMessage("视频播放中")

    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        if not self.is_camera_running:
            # 先停止视频播放（防止冲突）
            if self.is_video_playing:
                self.stop_detection()
                QThread.msleep(100)  # 短暂延迟确保资源释放

            # 开启摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "警告", "无法打开摄像头！")
                return

            # 更新状态和UI
            self.is_camera_running = True
            self.camera_btn.setText("📷 关闭摄像头")
            self.stop_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.video_info_label.setText("摄像头模式 - 检测中...")
            self.statusBar().showMessage("摄像头已开启，开始检测...")

            # 隐藏进度条（摄像头不需要）
            self.progress_panel.hide()

            # 启动摄像头检测定时器
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self.process_camera_frame)
            self.camera_timer.start(30)  # 约30fps
        else:
            # 关闭摄像头
            self._stop_camera()

    def _stop_camera(self):
        """停止摄像头"""
        self.stop_recording()
        self.is_camera_running = False
        
        # 释放资源
        if self.cap:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'camera_timer'):
            self.camera_timer.stop()
        
        # 重置UI
        self.camera_btn.setText("📷 开启摄像头")
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.record_btn.setText("⏺️ 开始录制")
        self.video_label.clear()
        self.video_label.setText("摄像头已关闭")
        self.video_info_label.setText("请选择检测模式")
        self.statusBar().showMessage("摄像头已关闭")

    def process_camera_frame(self):
        """处理摄像头帧 - 使用自定义绘制"""
        if not self.is_camera_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            # 更新FPS计算
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.camera_fps = self.frame_count / elapsed
                self.start_time = time.time()
                self.frame_count = 0

            # 获取置信度阈值
            conf_threshold = self.conf_slider.value() / 100.0

            # 进行检测
            results = self.model(frame, conf=conf_threshold, verbose=False)

            # 使用自定义绘制函数
            annotated_frame, helmet_count, no_helmet_count = draw_custom_boxes(
                frame, results, conf_threshold
            )

            # 录制视频
            if self.is_recording and self.video_writer:
                self.video_writer.write(annotated_frame)

            # 保存当前帧（用于显示和保存）
            self.current_frame = annotated_frame.copy()

            # 转换为QT图像格式并显示
            self.display_image(annotated_frame)

            # 更新统计信息
            self.fps_label.setText(f"FPS: {self.camera_fps:.1f}")
            total_objects = helmet_count + no_helmet_count
            self.objects_label.setText(f"检测对象: {total_objects}")
            self.helmet_label.setText(f"🔵 佩戴安全帽: {helmet_count}")
            self.no_helmet_label.setText(f"🔴 未戴安全帽: {no_helmet_count}")

    def select_image(self):
        """选择图片进行检测 - 使用自定义绘制"""
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"
        )

        if file_path:
            # 停止其他检测模式
            self.stop_detection()

            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "警告", "无法读取图片文件！")
                return

            # 更新UI状态
            self.save_btn.setEnabled(True)
            self.record_btn.setEnabled(False)
            self.progress_panel.hide()
            self.video_info_label.setText(f"图片模式: {os.path.basename(file_path)}")

            # 获取置信度阈值
            conf_threshold = self.conf_slider.value() / 100.0

            # 进行检测
            results = self.model(image, conf=conf_threshold, verbose=False)

            # 使用自定义绘制函数
            annotated_image, helmet_count, no_helmet_count = draw_custom_boxes(
                image, results, conf_threshold
            )

            # 保存当前帧
            self.current_frame = annotated_image.copy()

            # 显示结果
            self.display_image(annotated_image)

            # 更新统计信息
            total_objects = helmet_count + no_helmet_count
            self.objects_label.setText(f"检测对象: {total_objects}")
            self.helmet_label.setText(f"🔵 佩戴安全帽: {helmet_count}")
            self.no_helmet_label.setText(f"🔴 未戴安全帽: {no_helmet_count}")

            self.statusBar().showMessage(f"图片检测完成: {os.path.basename(file_path)}")

    def select_video(self):
        """选择视频进行检测"""
        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return

        # 先停止摄像头（防止冲突）
        if self.is_camera_running:
            self.toggle_camera()
            QThread.msleep(100)  # 短暂延迟确保资源释放

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*.*)"
        )

        if file_path:
            self.video_path = file_path
            self.is_video_playing = True
            self.is_video_paused = False
            self.video_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.play_pause_btn.setEnabled(True)
            self.play_pause_btn.setText("⏸️")

            # 显示进度条
            self.progress_panel.show()
            self.video_slider.setEnabled(True)

            # 停止现有的视频线程
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()

            # 启动新的视频检测线程
            self.video_thread = VideoThread(
                model=self.model,
                video_path=file_path,
                conf_threshold=self.conf_slider.value() / 100.0
            )
            self.video_thread.frame_processed.connect(self.display_video_frame)
            self.video_thread.progress_updated.connect(self.update_video_progress)
            self.video_thread.video_info_ready.connect(self.on_video_info_ready)
            self.video_thread.playback_state_changed.connect(self.on_playback_state_changed)
            self.video_thread.finished.connect(self.video_finished)
            self.video_thread.start()

            self.video_info_label.setText(f"视频模式: {os.path.basename(file_path)}")
            self.statusBar().showMessage("视频加载中...")

    def on_video_info_ready(self, total_frames, fps):
        """接收视频信息"""
        self.video_frames_total = total_frames
        self.video_fps = fps

        # 更新总时间显示
        total_time = self.format_time(total_frames / fps if fps > 0 else 0)
        self.time_label.setText(f"00:00 / {total_time}")

    def display_video_frame(self, frame, stats):
        """显示视频帧"""
        # 保存当前帧
        self.current_frame = frame.copy()

        # 录制视频
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

        self.display_image(frame)
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        self.objects_label.setText(f"检测对象: {stats['total']}")
        self.helmet_label.setText(f"🔵 佩戴安全帽: {stats['helmet']}")
        self.no_helmet_label.setText(f"🔴 未戴安全帽: {stats['no_helmet']}")

    def update_video_progress(self, current_frame, total_frames, fps):
        """更新视频进度条 - 修复版"""
        # 更新成员变量
        if total_frames > 0:
            self.video_frames_total = total_frames
            self.video_fps = fps

        self.video_current_frame = current_frame

        # 更新进度条（如果不是用户正在拖动）
        if not self.is_seeking and total_frames > 0:
            # 将当前帧转换为0-1000的范围
            progress = int((current_frame / total_frames) * 1000)
            self.video_slider.setValue(progress)

            # 更新时间显示
            current_time = self.format_time(current_frame / fps if fps > 0 else 0)
            total_time = self.format_time(total_frames / fps if fps > 0 else 0)
            self.time_label.setText(f"{current_time} / {total_time}")

    def format_time(self, seconds):
        """格式化时间显示"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def on_slider_pressed(self):
        """滑块被按下"""
        self.is_seeking = True

    def on_slider_moved(self, value):
        """滑块被移动（拖动中）"""
        if self.is_video_playing and self.video_frames_total > 0:
            # 更新显示的时间（但不实际跳转，等释放后再跳转）
            # 将0-1000的值转换回帧数
            target_frame = int((value / 1000.0) * self.video_frames_total)
            current_time = self.format_time(target_frame / self.video_fps if self.video_fps > 0 else 0)
            total_time = self.format_time(self.video_frames_total / self.video_fps if self.video_fps > 0 else 0)
            self.time_label.setText(f"{current_time} / {total_time}")

    def on_slider_released(self):
        """滑块被释放"""
        if self.is_video_playing and self.video_thread and self.video_thread.isRunning():
            # 获取滑块值并跳转（0-1000转换为帧数）
            value = self.video_slider.value()
            target_frame = int((value / 1000.0) * self.video_frames_total)
            self.video_thread.seek_to(target_frame)

        self.is_seeking = False

    def video_finished(self):
        """视频检测完成"""
        self.is_video_playing = False
        self.is_video_paused = False
        self.video_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.play_pause_btn.setEnabled(False)
        self.record_btn.setText("⏺️ 开始录制")
        self.stop_recording()
        self.video_info_label.setText("视频检测完成")
        self.statusBar().showMessage("视频检测完成")
        self.video_slider.setEnabled(False)

        # 确保进度条到达最后
        self.video_slider.setValue(1000)
        total_time = self.format_time(self.video_frames_total / self.video_fps if self.video_fps > 0 else 0)
        self.time_label.setText(f"{total_time} / {total_time}")

    def save_current(self):
        """保存当前检测结果"""
        if self.current_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的内容！")
            return

        # 生成保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 根据当前模式确定保存目录
        if self.is_camera_running:
            save_dir = "saved_results/camera"
            file_name = f"camera_{timestamp}.jpg"
        elif self.is_video_playing:
            save_dir = "saved_results/video_snapshots"
            file_name = f"video_snapshot_{timestamp}.jpg"
        else:
            save_dir = "saved_results/images"
            file_name = f"image_{timestamp}.jpg"

        save_path = os.path.join(save_dir, file_name)

        # 创建目录并保存图片
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_path, self.current_frame)

        # 更新状态和UI
        self.save_status_label.setText(f"已保存: {file_name}")
        self.statusBar().showMessage(f"图片已保存: {save_path}")

        # 添加到历史记录
        self.add_to_history(save_path, "图片")

        # 显示保存成功提示
        QMessageBox.information(self, "保存成功", f"图片已保存到:\n{save_path}")

    def toggle_recording(self):
        """切换录制状态"""
        if self.is_recording:
            self.stop_recording()
            self.record_btn.setText("⏺️ 开始录制")
            self.statusBar().showMessage("录制已停止")
        else:
            if self._start_recording():
                self.record_btn.setText("⏺️ 停止录制")
                self.statusBar().showMessage("开始录制...")

    def _start_recording(self):
        """开始录制视频"""
        if not self.is_camera_running and not self.is_video_playing:
            QMessageBox.warning(self, "警告", "只能在摄像头或视频模式下录制！")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 根据当前模式确定保存目录
        if self.is_camera_running:
            save_dir = "saved_results/camera_recordings"
            file_name = f"camera_recording_{timestamp}.mp4"
        else:  # is_video_playing
            save_dir = "saved_results/video_recordings"
            file_name = f"video_recording_{timestamp}.mp4"

        save_path = os.path.join(save_dir, file_name)

        # 获取帧尺寸并创建视频写入器
        if self.current_frame is not None:
            os.makedirs(save_dir, exist_ok=True)
            height, width = self.current_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(save_path, fourcc, 20.0, (width, height))
            self.is_recording = True

            # 添加到历史记录
            self.add_to_history(save_path, "视频")
            self.save_status_label.setText(f"录制中: {file_name}")
            return True
        
        return False

    def stop_recording(self):
        """停止录制"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.record_btn.setText("⏺️ 开始录制")

    def add_to_history(self, file_path, file_type):
        """添加到历史记录"""
        file_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%H:%M:%S")

        item_text = f"{timestamp} - {file_type}: {file_name}"
        self.history_list.insertItem(0, item_text)

        # 限制历史记录数量
        if self.history_list.count() > 10:
            self.history_list.takeItem(10)

    def open_saved_file(self, item):
        """打开保存的文件"""
        try:
            # 从项目文本中提取文件名
            item_text = item.text()
            file_name = item_text.split(": ")[-1]

            # 在所有保存目录中查找文件
            search_dirs = [
                "saved_results/camera",
                "saved_results/camera_recordings",
                "saved_results/images",
                "saved_results/video_snapshots",
                "saved_results/video_recordings"
            ]

            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for root, _, files in os.walk(search_dir):
                        if file_name in files:
                            file_path = os.path.join(root, file_name)
                            os.startfile(file_path)  # 在Windows中打开文件
                            self.statusBar().showMessage(f"打开文件: {file_path}")
                            return

            QMessageBox.warning(self, "警告", f"找不到文件: {file_name}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"打开文件失败: {str(e)}")

    def clear_history(self):
        """清空历史记录"""
        self.history_list.clear()
        self.statusBar().showMessage("历史记录已清空")

    def stop_detection(self):
        """停止当前检测"""
        # 停止摄像头
        if self.is_camera_running:
            self._stop_camera()

        # 停止视频
        if self.is_video_playing and self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None
            self.video_finished()

        # 停止录制
        self.stop_recording()

        # 清理UI
        self.video_label.clear()
        self.video_label.setText("检测已停止")
        self.video_info_label.setText("请选择检测模式")
        self.save_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.play_pause_btn.setEnabled(False)
        self.progress_panel.hide()

    def display_image(self, cv_image):
        """将OpenCV图像转换为QT图像并显示"""
        # 转换颜色空间 BGR -> RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # 获取图像尺寸
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # 创建QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放以适应标签
        scaled_image = qt_image.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # 显示图像
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def closeEvent(self, event):
        """关闭窗口时的清理工作"""
        # 停止所有检测
        self.stop_detection()

        # 保存历史记录到文件
        self.save_history()

        event.accept()

    def save_history(self):
        """保存历史记录到文件"""
        try:
            os.makedirs("saved_results", exist_ok=True)
            with open("saved_results/history.txt", "w", encoding="utf-8") as f:
                for i in range(self.history_list.count()):
                    f.write(self.history_list.item(i).text() + "\n")
        except:
            pass


# ============ 主程序入口 ============
def main():
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 设置全局字体
    font = QFont("微软雅黑", 10)
    app.setFont(font)

    # 创建并显示主窗口
    window = HelmetDetectionApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
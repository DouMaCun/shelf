# -*- coding: utf-8 -*-
"""
视频输入源抽象层。

功能：
  提供统一的视频输入接口，支持三种输入源：
  1. 本地视频文件 (FileSource)
  2. USB 摄像头 (USBSource)
  3. RTSP 网络摄像头 (RTSPSource)

设计思路：
  - 每种输入源继承 VideoSource 基类，只需实现 _open() 和 _read_frame() 两个方法。
  - 使用后台线程异步读取帧，存入固定大小的双端队列缓冲区。
  - 通过目标帧率控制采样频率，避免不必要的重复处理。
  - 主线程通过 read() 方法非阻塞地从缓冲区取帧。

使用示例:
  >>> cfg = {"type": "file", "source": "demo.mp4", "fps": 5}
  >>> src = create_source(cfg)
  >>> src.start()
  >>> frame = src.read()
  >>> src.stop()
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


# ============================================================================
# 视频源基类
# ============================================================================

class VideoSource(ABC):
    """视频源抽象基类。

    封装了帧读取线程、缓冲区管理和帧率控制的通用逻辑。
    子类只需实现 _open() 和 _read_frame() 两个方法即可接入不同的视频源。

    Attributes:
        source: 视频源地址（文件路径、摄像头索引、RTSP URL）
        target_fps: 目标帧率，后台线程按此频率采样
        width: 处理后帧的宽度（像素）
        height: 处理后帧的高度（像素）
        buffer: 帧缓冲区（双端队列），主线程通过 read() 从左侧取帧
        _running: 后台线程运行标志
        _total_frames: 已读取的总帧数（用于统计实际帧率）
    """

    def __init__(self, source: str, fps: int = 5, width: int = 1280, height: int = 720,
                 buffer_size: int = 30):
        """
        Args:
            source: 视频源地址
            fps: 目标帧率，默认 5fps（货架场景变化较慢，足够用）
            width: 帧宽度
            height: 帧高度
            buffer_size: 帧缓冲队列大小，默认 30 帧
        """
        self.source = source
        self.target_fps = fps
        self.width = width
        self.height = height
        # 双端队列作为帧缓冲区：主线程从左侧取，后台线程从右侧放
        self.buffer = deque(maxlen=buffer_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_frame_time = 0.0
        self._frame_interval = 1.0 / max(fps, 1)  # 计算帧间隔（秒）
        self._total_frames = 0
        self._start_time = 0.0

    # ---- 子类必须实现的方法 ----

    @abstractmethod
    def _open(self) -> cv2.VideoCapture:
        """打开视频源，返回 cv2.VideoCapture 对象。

        子类在此方法中完成：
        - 创建 cv2.VideoCapture
        - 设置分辨率、帧率等参数
        - 返回可用的 VideoCapture 实例
        """
        ...

    @abstractmethod
    def _read_frame(self) -> np.ndarray | None:
        """从视频源读取一帧。

        子类实现具体的帧读取逻辑（如 RTSP 可能涉及断线重连）。
        返回 None 表示视频流结束或读取失败。
        """
        ...

    # ---- 公共接口 ----

    def start(self) -> None:
        """启动后台帧读取线程。

        执行步骤：
        1. 调用子类的 _open() 打开视频源
        2. 启动后台线程，循环调用 _read_frame()
        3. 后台线程按目标帧率控制读取速度，帧缩放后放入缓冲区

        Raises:
            RuntimeError: 视频源无法打开
        """
        if self._running:
            return  # 避免重复启动

        # 打开视频源
        self._capture = self._open()
        if not self._capture.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        self._running = True
        self._start_time = time.time()
        # daemon=True 确保主线程退出时自动清理
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"视频源已启动: type={self.__class__.__name__}, source={self.source}, "
                    f"fps={self.target_fps}, {self.width}x{self.height}")

    def stop(self) -> None:
        """停止帧读取线程并释放所有资源。

        执行步骤：
        1. 设置 _running = False 通知后台线程退出
        2. 等待后台线程结束（最多 5 秒超时）
        3. 释放 cv2.VideoCapture 资源
        4. 清空帧缓冲区
        5. 输出帧读取统计
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if hasattr(self, '_capture'):
            self._capture.release()
        self.buffer.clear()
        elapsed = time.time() - self._start_time
        logger.info(f"视频源已停止: 读取 {self._total_frames} 帧, "
                    f"实际帧率 {self._total_frames / max(elapsed, 0.01):.1f}")

    def read(self) -> np.ndarray | None:
        """从缓冲区取一帧（非阻塞，线程安全）。

        使用 deque.popleft() 从左侧取出最旧的帧，
        如果缓冲区为空则返回 None。

        Returns:
            np.ndarray | None: BGR 格式的帧图像，缓冲区空时返回 None
        """
        try:
            return self.buffer.popleft()
        except IndexError:
            return None

    @property
    def actual_fps(self) -> float:
        """实际帧率（从启动到现在的平均值）。

        Returns:
            float: 实际帧率 = 总帧数 / 运行时间
        """
        elapsed = time.time() - self._start_time
        return self._total_frames / max(elapsed, 0.01)

    # ---- 内部实现 ----

    def _read_loop(self) -> None:
        """后台线程主循环。

        工作流程：
        1. 检查当前时间与上次读取时间的间隔
        2. 如果间隔 >= 帧间隔，则读取一帧
        3. 对帧进行缩放（按 width/height）
        4. 将缩放后的帧放入缓冲区
        5. 如果读取失败（视频结束），自动停止

        注意：sleep(0.001) 用于避免空转时消耗过多 CPU。
        """
        while self._running:
            now = time.time()

            # 帧率控制：如果距离上次读取还不满一个帧间隔，等待
            if now - self._last_frame_time < self._frame_interval:
                time.sleep(0.001)
                continue

            # 读取帧
            frame = self._read_frame()
            if frame is None:
                logger.warning("视频流结束或读取失败")
                self._running = False
                break

            # 缩放到目标分辨率
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            # 放入缓冲区（右侧追加，当缓冲区满时自动丢弃最旧的帧）
            self.buffer.append(frame)
            self._total_frames += 1
            self._last_frame_time = now


# ============================================================================
# 具体视频源实现
# ============================================================================

class FileSource(VideoSource):
    """本地视频文件源。

    适用于离线测试和回放场景。支持 mp4、avi、mkv 等常见格式。
    """

    def _open(self) -> cv2.VideoCapture:
        """打开本地视频文件。

        会检查文件是否存在，不存在时抛出 FileNotFoundError。

        Returns:
            cv2.VideoCapture: 指向视频文件的捕获对象
        """
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.source}")
        return cv2.VideoCapture(str(path))

    def _read_frame(self) -> np.ndarray | None:
        """从文件读取下一帧。

        当 opencv 返回 ret=False 时表示文件播放完毕，返回 None。
        """
        ret, frame = self._capture.read()
        return frame if ret else None


class USBSource(VideoSource):
    """USB 摄像头源。

    通过 DirectShow (Windows) 接口访问本地摄像头。
    适用于开发阶段使用本地摄像头模拟货架场景。
    """

    def __init__(self, source: str = "0", **kwargs):
        """
        Args:
            source: 摄像头索引（字符串形式的数字，如 "0"、"1"）
        """
        super().__init__(source, **kwargs)

    def _open(self) -> cv2.VideoCapture:
        """打开 USB 摄像头。

        使用 cv2.CAP_DSHOW 后端（Windows DirectShow），
        并尝试设置目标分辨率和帧率。
        """
        idx = int(self.source) if self.source.isdigit() else 0
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        # 设置摄像头参数（不保证所有摄像头都支持）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        return cap

    def _read_frame(self) -> np.ndarray | None:
        """从 USB 摄像头读取帧。"""
        ret, frame = self._capture.read()
        return frame if ret else None


class RTSPSource(VideoSource):
    """RTSP 网络摄像头源。

    适用于生产环境中的 IP 摄像头接入。
    内置断线重连机制：读取失败时自动重试连接。
    """

    def _open(self) -> cv2.VideoCapture:
        """打开 RTSP 流。

        使用 cv2.CAP_FFMPEG 后端处理 RTSP 协议。
        设置 BUFFERSIZE=1 以减少延迟（过大的缓冲会引入延迟）。
        """
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_frame(self) -> np.ndarray | None:
        """从 RTSP 流读取帧，读取失败时自动重连。

        RTSP 网络流可能因网络波动中断，这里做了一次简单的重连尝试：
        - 读取失败时释放旧的 VideoCapture
        - 等待 2 秒后重新连接
        - 再次尝试读取
        """
        ret, frame = self._capture.read()
        if not ret:
            logger.warning("RTSP 读取失败，尝试重连...")
            self._capture.release()
            time.sleep(2)
            self._capture = self._open()
            ret, frame = self._capture.read()
        return frame if ret else None


# ============================================================================
# 工厂函数
# ============================================================================

def create_source(config: dict) -> VideoSource:
    """工厂函数：根据配置字典创建对应的视频源实例。

    根据配置中的 type 字段选择不同的视频源实现：
    - "file": 本地视频文件
    - "usb": USB 摄像头
    - "rtsp": RTSP 网络摄像头

    Args:
        config: 视频输入配置字典，至少包含 {"type": "...", "source": "..."}

    Returns:
        VideoSource: 对应类型的视频源实例

    Raises:
        ValueError: 如果 type 不在支持的列表中

    Example:
        >>> cfg = {"type": "file", "source": "data/raw/demo.mp4", "fps": 5,
        ...        "width": 1280, "height": 720}
        >>> src = create_source(cfg)
    """
    source_type = config.get("type", "file")
    source_addr = config.get("source", "")

    # 提取通用参数
    kwargs = {
        "fps": config.get("fps", 5),
        "width": config.get("width", 1280),
        "height": config.get("height", 720),
        "buffer_size": config.get("buffer_size", 30),
    }

    if source_type == "file":
        return FileSource(source_addr, **kwargs)
    elif source_type == "usb":
        return USBSource(source_addr, **kwargs)
    elif source_type == "rtsp":
        return RTSPSource(source_addr, **kwargs)
    else:
        raise ValueError(f"不支持的视频源类型: {source_type}")

# -*- coding: utf-8 -*-
"""
帧预处理工具。

功能：
  提供 YOLO 模型推理前所需的帧预处理操作，包括：
  1. 缩放（resize）
  2. 归一化（normalize）
  3. 颜色空间转换（BGR -> RGB）
  4. CHW 布局转换（HWC -> CHW）
  5. letterbox 填充（保持宽高比的缩放方法）

设计思路：
  - 每个函数都是纯函数（无副作用），输入 numpy 数组，输出 numpy 数组。
  - 避免在管线主循环中做不必要的复制，尽量使用 opencv 和 numpy 的零拷贝操作。
  - letterbox 用于需要保持宽高比的场景（如原始 YOLO 推理），
    但在本项目中使用简单的 resize + 坐标映射即可（检测器内部会做坐标还原）。
"""

from __future__ import annotations

import cv2
import numpy as np


def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """缩放帧到指定尺寸。

    Args:
        frame: BGR 图像 (H, W, 3)
        width: 目标宽度
        height: 目标高度

    Returns:
        缩放后的图像 (height, width, 3)
    """
    return cv2.resize(frame, (width, height))


def normalize(frame: np.ndarray) -> np.ndarray:
    """像素值归一化到 [0, 1] 区间。

    将 uint8 图像（0-255）转为 float32 图像（0.0-1.0）。

    Args:
        frame: BGR 图像 (H, W, 3)，dtype uint8 或 float32

    Returns:
        float32 归一化图像
    """
    return frame.astype(np.float32) / 255.0


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """颜色空间转换：BGR -> RGB。

    OpenCV 默认使用 BGR 色彩顺序，而 YOLO 等深度学习模型通常使用 RGB。

    Args:
        frame: BGR 图像 (H, W, 3)

    Returns:
        RGB 图像 (H, W, 3)
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preprocess_for_yolo(frame: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """YOLO 模型完整预处理流水线。

    组合四个步骤：
    1. Resize 到模型输入尺寸
    2. BGR -> RGB 颜色空间转换
    3. 像素值归一化到 [0, 1]
    4. HWC -> CHW 布局转换
    5. 添加 batch 维度 (1, C, H, W)

    Args:
        frame: BGR 图像 (H, W, 3)
        input_size: 模型输入尺寸 (width, height)，通常为 (640, 640)

    Returns:
        np.ndarray: shape (1, 3, H, W), dtype float32, 值域 [0, 1]
                    可直接传入 ONNX Runtime 的 session.run()

    Example:
        >>> blob = preprocess_for_yolo(frame, (640, 640))
        >>> outputs = session.run([output_name], {input_name: blob})
    """
    # Step 1: 缩放到模型输入尺寸
    img = cv2.resize(frame, input_size)
    # Step 2: BGR -> RGB
    img = bgr_to_rgb(img)
    # Step 3: 归一化 [0, 255] -> [0, 1]
    img = normalize(img)
    # Step 4: HWC -> CHW（numpy transpose）
    img = np.transpose(img, (2, 0, 1))
    # Step 5: 添加 batch 维度
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def letterbox(frame: np.ndarray, target_size: tuple[int, int],
              color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, int, int]:
    """带填充的等比缩放（letterbox）。

    保持原始宽高比的情况下缩放图像，不足的部分用灰色填充。
    常用于需要保持物体形状不变的检测场景。

    工作流程：
    1. 计算缩放比例 scale，使长边恰好等于目标尺寸
    2. 按比例缩放图像
    3. 创建目标尺寸的灰色画布
    4. 将缩放后的图像居中放置在画布上

    Args:
        frame: BGR 输入图像 (H, W, 3)
        target_size: 目标尺寸 (width, height)
        color: 填充颜色 (B, G, R)，默认灰色 (114, 114, 114)

    Returns:
        tuple:
            - padded_img: 填充后的图像 (th, tw, 3)
            - scale: 缩放比例
            - pad_left: 左侧填充宽度（用于坐标映射）
            - pad_top: 顶部填充高度（用于坐标映射）

    Example:
        >>> padded, scale, pad_left, pad_top = letterbox(frame, (640, 640))
        >>> # 还原检测坐标时: x_orig = (x_padded - pad_left) / scale
    """
    h, w = frame.shape[:2]
    tw, th = target_size

    # 计算缩放比例：选较小的缩放因子，保证图像不会超出目标尺寸
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    # 缩放图像
    resized = cv2.resize(frame, (nw, nh))

    # 创建目标尺寸的灰色画布，将缩放后的图像居中放置
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    pad_left = (tw - nw) // 2
    pad_top = (th - nh) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    return canvas, scale, pad_left, pad_top

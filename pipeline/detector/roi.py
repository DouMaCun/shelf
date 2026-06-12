# -*- coding: utf-8 -*-
"""
ROI（Region of Interest）感兴趣区域处理。

功能：
  在货架监控场景中，并非整帧画面都需要推理。
  通常摄像头画面包含天花板、地面、过道等无关区域，真正的货架只占画面的一部分。
  通过定义 ROI 区域裁剪，可以：
  1. 减少送入模型的像素量，加速推理
  2. 排除背景区域的误检
  3. 聚焦真正的货架商品区域

坐标约定：
  配置文件中的 ROI 使用归一化坐标（0.0 ~ 1.0），表示相对于全帧的比例。
  这样当摄像头分辨率变化时，无需修改配置。
  例如: [0.0, 0.2, 1.0, 0.9] 表示画面左起 0%，上起 20% 到右起 100%，下起 90% 的矩形区域。
"""

from __future__ import annotations

import numpy as np


def apply_roi(frame: np.ndarray, roi: list[float]) -> np.ndarray:
    """根据归一化 ROI 坐标裁剪帧。

    Args:
        frame: BGR 输入图像 (H, W, 3)
        roi: 归一化坐标 [x1, y1, x2, y2]，范围 0.0 ~ 1.0

    Returns:
        np.ndarray: 裁剪后的子图像，尺寸 = (roi_h, roi_w, 3)

    Example:
        >>> # 裁剪画面中间的货架区域
        >>> roi_frame = apply_roi(frame, [0.1, 0.2, 0.9, 0.8])
    """
    h, w = frame.shape[:2]
    # 归一化坐标 -> 像素坐标，并用 max/min 防止越界
    x1 = max(0, int(roi[0] * w))
    y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w))
    y2 = min(h, int(roi[3] * h))
    return frame[y1:y2, x1:x2]


def roi_to_pixel(roi: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """将归一化 ROI 坐标转为像素坐标。

    配合 apply_roi 使用，当需要在原始帧上绘制标注或还原坐标时，
    需要知道 ROI 区域在原始帧中的偏移量。

    Args:
        roi: 归一化坐标 [x1, y1, x2, y2]
        width: 原始帧宽度
        height: 原始帧高度

    Returns:
        tuple: (x1, y1, x2, y2) 像素坐标
    """
    x1 = max(0, int(roi[0] * width))
    y1 = max(0, int(roi[1] * height))
    x2 = min(width, int(roi[2] * width))
    y2 = min(height, int(roi[3] * height))
    return x1, y1, x2, y2


def adjust_detection_to_full_frame(
    detections: np.ndarray,
    roi: list[float],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """将 ROI 区域内的检测坐标映射回全帧坐标。

    当使用 ROI 裁剪后进行检测时，检测框坐标是相对于 ROI 子图的。
    此函数将这些坐标还原为相对于全帧的坐标，便于后续的跟踪和可视化。

    还原公式: x_full = x_roi + roi_offset_x

    Args:
        detections: 检测结果，shape (N, 6)，[x1, y1, x2, y2, conf, class_id]
                    注意这里的坐标是相对于 ROI 子图的
        roi: ROI 归一化坐标 [rx1, ry1, rx2, ry2]
        frame_width: 全帧宽度
        frame_height: 全帧高度

    Returns:
        np.ndarray: shape (N, 6)，坐标已映射到全帧空间

    Example:
        >>> roi_frame = apply_roi(frame, [0.1, 0.2, 0.9, 0.8])
        >>> dets = detector.detect(roi_frame)  # 坐标相对于 roi_frame
        >>> dets_full = adjust_detection_to_full_frame(dets, [0.1, 0.2, 0.9, 0.8], 1920, 1080)
    """
    rx1, ry1, _, _ = roi_to_pixel(roi, frame_width, frame_height)

    # 复制检测结果，避免修改原数组
    dets = detections.copy()
    # 将 ROI 子图坐标加上 ROI 区域在全帧中的偏移量
    dets[:, 0] += rx1   # x1
    dets[:, 1] += ry1   # y1
    dets[:, 2] += rx1   # x2
    dets[:, 3] += ry1   # y2
    return dets

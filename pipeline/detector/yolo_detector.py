# -*- coding: utf-8 -*-
"""
YOLO 目标检测器封装。

功能：
  基于 ONNX Runtime 的 YOLO 目标检测器，支持 YOLOv8/v11 等主流 YOLO 模型。
  将 PyTorch 模型导出为 ONNX 格式后，通过此检测器进行高性能推理。

核心职责：
  1. 加载 ONNX 模型并创建推理会话
  2. 将原始帧预处理为模型输入张量
  3. 执行推理并解析输出：坐标解码、NMS、坐标还原
  4. 输出标准化检测框 [x1, y1, x2, y2, confidence, class_id]

设计要点：
  - 支持两种 YOLO ONNX 输出格式：
    a. (1, 84, 8400) — 4 个坐标通道 + 80 个类别通道，在 8400 个锚点上
    b. (1, N, 6) — 每行 [x1, y1, x2, y2, conf, class_id]，直接可用
  - NMS 使用纯 numpy 实现，不依赖 torch/torchvision，保持轻量化
  - 坐标映射：模型在 (640, 640) 输入上推理，输出坐标已映射回原始帧尺寸
  - 内置推理耗时统计，用于性能监控

使用示例:
  >>> detector = YOLODetector("models/yolo11n.onnx", conf_threshold=0.5)
  >>> detector.load()
  >>> detections = detector.detect(frame)  # shape (N, 6)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from loguru import logger

from pipeline.io.preprocess import preprocess_for_yolo


class YOLODetector:
    """YOLO 目标检测器（ONNX Runtime 推理引擎）。

    封装了模型加载、预处理、推理、后处理的完整流程。

    Attributes:
        model_path: ONNX 模型文件路径
        conf_threshold: 置信度阈值，低于此值的检测结果将被过滤
        iou_threshold: NMS 的 IoU 阈值，高于此值的重叠框将被抑制
        input_size: 模型输入尺寸 (width, height)，从 ONNX 模型定义中获取
        _session: ONNX Runtime 推理会话
        _inference_times: 最近 50 次推理耗时列表（毫秒），用于性能统计
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
    ):
        """
        Args:
            model_path: ONNX 模型文件路径
            conf_threshold: 置信度阈值，默认 0.5
            iou_threshold: NMS IoU 阈值，默认 0.45（YOLO 标准值）
            input_size: 模型输入尺寸，默认 (640, 640)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX 模型文件不存在: {model_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # ONNX Runtime 推理会话（懒加载，首帧推理时初始化）
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._output_name: str = ""

        # 推理耗时滑动窗口（保留最近 50 次）
        self._inference_times: list[float] = []

    # ========================================================================
    # 模型加载
    # ========================================================================

    def load(self) -> None:
        """加载 ONNX 模型并创建推理会话。

        检查可用的推理后端（CUDA、TensorRT、OpenVINO、CPU 等），
        优先使用 GPU 后端，无 GPU 时回退到 CPU。

        工作流程：
        1. 查询 ONNX Runtime 可用的执行提供者（Execution Providers）
        2. 创建 InferenceSession，指定优先级最高的提供者
        3. 获取输入/输出节点名称和形状信息
        """
        providers = ort.get_available_providers()
        logger.info(f"可用推理后端: {providers}")

        # 创建推理会话，让 ONNX Runtime 自动选择最优后端
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        # 获取输入/输出节点名称（用于 session.run() 时指定张量）
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # 输出模型信息以便调试
        input_shape = self._session.get_inputs()[0].shape
        output_shape = self._session.get_outputs()[0].shape
        logger.info(f"模型已加载: {self.model_path}, "
                    f"input={input_shape}, output={output_shape}")

    # ========================================================================
    # 推理接口
    # ========================================================================

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """对单帧图像执行目标检测。

        完整的推理流水线：预处理 -> 推理 -> 后处理 -> 输出。

        Args:
            frame: BGR 图像 (H, W, 3)，dtype uint8

        Returns:
            np.ndarray: 检测结果，shape (N, 6)，每列含义：
                [0] x1 — 左上角 x 坐标（原始帧像素）
                [1] y1 — 左上角 y 坐标（原始帧像素）
                [2] x2 — 右下角 x 坐标（原始帧像素）
                [3] y2 — 右下角 y 坐标（原始帧像素）
                [4] confidence — 检测置信度 [0, 1]
                [5] class_id — 类别 ID（整数，对应 model_mapping.yaml 中的映射）
                如果未检测到任何目标，返回 shape (0, 6) 的空数组
        """
        if self._session is None:
            self.load()

        # 记录原始帧尺寸（后处理时用于坐标映射）
        orig_h, orig_w = frame.shape[:2]

        # ---- 预处理：BGR帧 -> ONNX输入张量 ----
        blob = preprocess_for_yolo(frame, self.input_size)

        # ---- 推理 ----
        t0 = time.perf_counter()
        outputs = self._session.run([self._output_name], {self._input_name: blob})
        elapsed = time.perf_counter() - t0
        self._inference_times.append(elapsed)

        # ---- 后处理：原始输出 -> 标准化检测框 ----
        detections = self._postprocess(outputs[0], orig_w, orig_h)

        return detections

    # ========================================================================
    # 后处理
    # ========================================================================

    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """YOLO 模型输出后处理流水线。

        三个核心步骤：
        1. 解析输出格式：处理两种常见的 YOLO ONNX 输出格式
        2. NMS 非极大值抑制：去除重叠的冗余检测框
        3. 坐标映射：将模型空间坐标映射回原始帧像素坐标

        Args:
            output: 模型原始输出张量（来自 ONNX Runtime）
            orig_w: 原始帧宽度
            orig_h: 原始帧高度

        Returns:
            np.ndarray: shape (N, 6), 标准化检测框
        """
        # ---- Step 1: 解析输出格式 ----

        # 如果是 3 维张量（如 (1, 84, 8400)），去 batch 维度
        if output.ndim == 3:
            output = output[0]  # -> (84, 8400)

        # 格式 A: (6, N) 即转置后的 (N, 6)
        # 判断逻辑：如果第一维是 6（x1,y1,x2,y2,conf,cls），且第二维远大于 6
        if output.shape[0] == 6 and output.shape[1] > 6:
            output = output.T  # -> (N, 6)

        # 格式 B: (N, 4+num_classes) 即 anchor 格式，需要解码
        if output.shape[1] > 6:
            # 前 4 列是边界框坐标，后面是每个类别的置信度
            boxes = output[:, :4]      # (N, 4)
            scores = output[:, 4:]     # (N, num_classes)

            # 取每个锚点的最大置信度和对应类别
            class_ids = np.argmax(scores, axis=1)   # (N,)
            confs = np.max(scores, axis=1)          # (N,)

            # 按置信度阈值过滤
            mask = confs >= self.conf_threshold
            boxes = boxes[mask]
            confs = confs[mask]
            class_ids = class_ids[mask]

            # 坐标格式转换：cxcywh -> xyxy
            # YOLO 模型通常输出中心点 + 宽高，需要转为左上角 + 右下角
            boxes = self._cxcywh_to_xyxy(boxes)
        else:
            # 格式 C: (N, 6) 已解码，直接取
            mask = output[:, 4] >= self.conf_threshold
            output = output[mask]
            boxes = output[:, :4]
            confs = output[:, 4]
            class_ids = output[:, 5].astype(int)

        # 如果所有检测框都被阈值过滤，返回空数组
        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # ---- Step 2: NMS 非极大值抑制 ----
        # 去除高度重叠的冗余框，只保留每个目标的最佳框
        indices = self._nms(boxes, confs, self.iou_threshold)
        boxes = boxes[indices]
        confs = confs[indices]
        class_ids = class_ids[indices]

        # ---- Step 3: 坐标映射（模型空间 -> 原始帧空间） ----
        # 模型输入尺寸是 (640, 640)，检测坐标在模型空间中
        # 需要按比例映射回原始帧尺寸
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        boxes[:, [0, 2]] *= scale_x   # x 坐标
        boxes[:, [1, 3]] *= scale_y   # y 坐标

        # 将坐标裁剪到图像边界内（防止越界）
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # ---- 组装最终输出 ----
        # 格式: [x1, y1, x2, y2, conf, class_id]
        detections = np.column_stack([boxes, confs, class_ids.astype(np.float32)])
        return detections.astype(np.float32)

    # ========================================================================
    # 静态工具方法
    # ========================================================================

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """边界框坐标格式转换：cxcywh -> xyxy。

        YOLO 模型原始输出使用中心点坐标格式：
        [center_x, center_y, width, height]

        转换为更方便的左上-右下格式：
        [x1, y1, x2, y2]

        转换公式：
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        """
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """纯 numpy 实现的非极大值抑制（NMS）。

        算法流程：
        1. 按置信度降序排列所有检测框
        2. 取置信度最高的框 A，加入保留列表
        3. 计算 A 与剩余所有框的 IoU
        4. 将 IoU > iou_threshold 的框视为与 A 重叠的重复框，移除
        5. 对剩余的框重复步骤 2-4

        Args:
            boxes: 检测框数组 (M, 4)，每行 [x1, y1, x2, y2]
            scores: 置信度数组 (M,)
            iou_threshold: IoU 阈值，超过此值的框被视为重复

        Returns:
            np.ndarray: 保留的检测框索引数组
        """
        # 按置信度降序排序
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            # 取当前置信度最高的框
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # 计算当前框与剩余所有框的 IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            # 交集面积
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            # 并集面积
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                          (boxes[order[1:], 3] - boxes[order[1:], 1])

            # IoU = 交集 / 并集
            iou = inter / np.maximum(area_i + area_others - inter, 1e-6)

            # 保留 IoU 低于阈值的框（不重复的框）
            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return np.array(keep, dtype=int)

    # ========================================================================
    # 性能统计
    # ========================================================================

    @property
    def avg_inference_time(self) -> float:
        """平均推理耗时（秒）。

        取最近 50 次推理的滑动平均，避免瞬时波动影响判断。

        Returns:
            float: 平均推理时间（秒），如 0.015 表示约 15ms
        """
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times[-50:]) / len(self._inference_times[-50:])

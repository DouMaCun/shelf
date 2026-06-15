# -*- coding: utf-8 -*-
"""
RKNN NPU 目标检测器（运行在 RK3588 设备上）。

功能：
  使用 rknn-toolkit-lite2 在 RK3588 NPU 上运行 YOLO 推理。
  对外接口与 YOLODetector 完全相同，管线代码无需改动。

依赖：
  设备上安装 rknn-toolkit-lite2：
    pip install rknn-toolkit-lite2

使用前提：
  .rknn 模型由开发机上的 tools/model_export_rknn.py 转换生成。

与 YOLODetector 的区别：
  - 模型文件格式：.rknn（非 .onnx）
  - 推理库：rknnlite（非 onnxruntime）
  - 推理后端：NPU（非 CPU）
  - 后处理：复用 YOLODetector 的 _cxcywh_to_xyxy / _nms，保持一致
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from pipeline.io.preprocess import bgr_to_rgb
from pipeline.detector.yolo_detector import YOLODetector


class RKNNDetector:
    """RKNN NPU 目标检测器。

    接口与 YOLODetector 完全相同：
      detector = RKNNDetector("models/yolo11n.rknn", conf_threshold=0.5)
      detector.load()
      detections = detector.detect(frame)  # ndarray (N, 6)

    Attributes:
        model_path: .rknn 模型文件路径
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值
        input_size: 模型输入尺寸 (width, height)
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN 模型文件不存在: {model_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        self._rknn: object | None = None
        self._inference_times: list[float] = []

    # ========================================================================
    # 模型加载
    # ========================================================================

    def load(self) -> None:
        """加载 RKNN 模型，初始化 NPU 推理上下文。

        Raises:
            ImportError: 设备上未安装 rknn-toolkit-lite2
            RuntimeError: 模型加载或 NPU 初始化失败
        """
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise ImportError(
                "未找到 rknn-toolkit-lite2，请在 RK3588 设备上安装：\n"
                "  pip install rknn-toolkit-lite2\n"
                "注意：此库只能在 RK3588/RK3566 等 Rockchip 设备上使用。"
            )

        self._rknn = RKNNLite(verbose=False)

        ret = self._rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 模型失败，错误码: {ret}")

        # 初始化 NPU 运行时（core_mask=RKNNLite.NPU_CORE_AUTO 自动分配核心）
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"NPU 初始化失败，错误码: {ret}")

        logger.info(f"RKNN 模型已加载: {self.model_path}")

    # ========================================================================
    # 推理接口（与 YOLODetector.detect 完全相同的签名）
    # ========================================================================

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """对单帧图像执行目标检测。

        Args:
            frame: BGR 图像 (H, W, 3)，dtype uint8

        Returns:
            np.ndarray: shape (N, 6)，[x1, y1, x2, y2, confidence, class_id]
                        坐标为原始帧像素坐标。无检测结果返回 shape (0, 6)。
        """
        if self._rknn is None:
            self.load()

        orig_h, orig_w = frame.shape[:2]

        # ---- 预处理：BGR → RGB → resize → NHWC uint8 ----
        img = cv2.resize(frame, self.input_size)
        img = bgr_to_rgb(img)
        # RKNN 默认接收 uint8 NHWC 输入（不需要归一化，模型转换时已配置 mean/std）
        blob = np.expand_dims(img, axis=0)   # (1, H, W, 3)

        # ---- 推理 ----
        t0 = time.perf_counter()
        outputs = self._rknn.inference(inputs=[blob])
        elapsed = time.perf_counter() - t0
        self._inference_times.append(elapsed)

        # ---- 后处理 ----
        return self._postprocess(outputs[0], orig_w, orig_h)

    # ========================================================================
    # 后处理（复用 YOLODetector 的静态方法）
    # ========================================================================

    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """RKNN 输出后处理，与 YOLODetector._postprocess 逻辑对齐。

        RKNN 输出格式与 ONNX 相同（转换时保留了输出结构），
        直接复用 ONNX 后处理逻辑。
        """
        # 去掉 batch 维度
        if output.ndim == 3:
            output = output[0]

        # 格式 A: (6, N) 转置
        if output.shape[0] == 6 and output.shape[1] > 6:
            output = output.T

        # 格式 B: (N, 4+num_classes) anchor 格式解码
        if output.shape[1] > 6:
            boxes = output[:, :4]
            scores = output[:, 4:]
            class_ids = np.argmax(scores, axis=1)
            confs = np.max(scores, axis=1)
            mask = confs >= self.conf_threshold
            boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
            boxes = YOLODetector._cxcywh_to_xyxy(boxes)
        else:
            # 格式 C: (N, 6) 已解码
            mask = output[:, 4] >= self.conf_threshold
            output = output[mask]
            boxes = output[:, :4]
            confs = output[:, 4]
            class_ids = output[:, 5].astype(int)

        if len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # NMS
        indices = YOLODetector._nms(boxes, confs, self.iou_threshold)
        boxes, confs, class_ids = boxes[indices], confs[indices], class_ids[indices]

        # 坐标映射：模型空间 → 原始帧空间
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * scale_x, 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * scale_y, 0, orig_h)

        return np.column_stack([boxes, confs, class_ids.astype(np.float32)]).astype(np.float32)

    # ========================================================================
    # 性能统计（与 YOLODetector 相同的接口）
    # ========================================================================

    @property
    def avg_inference_time(self) -> float:
        """平均推理耗时（秒），取最近 50 次的滑动平均。"""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times[-50:]) / len(self._inference_times[-50:])

    def release(self) -> None:
        """释放 NPU 资源。服务关闭时调用。"""
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None
            logger.info("RKNN NPU 资源已释放")

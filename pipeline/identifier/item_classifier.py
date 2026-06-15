# -*- coding: utf-8 -*-
"""
基于 YOLO 的商品视觉分类器。

功能：
  在条码扫描失败时，通过 YOLO 对俯拍图像做商品分类，识别 SKU。
  与 BarcodeReader 配合使用，构成"条码优先、YOLO 回退"的双轨识别策略。

设计说明：
  - 复用 YOLODetector（或 RKNNDetector）的 detect() 接口，不重新实现推理
  - 从所有检测框中取置信度最高的作为结果
  - 通过 model_mapping.yaml 将 class_id 映射为 sku_id
  - 商品未识别（无检测框 / 置信度不足）时返回 (None, 0.0)

注意：
  此分类器使用的 YOLO 模型应是**针对俯拍角度定制训练**的模型，
  而非通用 COCO 预训练模型（COCO 模型用于手臂检测）。
  两个模型路径在 config/default.yaml 中分别配置：
    detector.model_path       → 手臂/人体检测（COCO 预训练）
    item_classifier.model_path → 商品分类（自定义训练）
"""

from __future__ import annotations

import yaml
import numpy as np
from loguru import logger


class ItemClassifier:
    """基于 YOLO 检测器的商品分类器。

    接收一帧图像，输出最可能的 (sku_id, confidence) 对。

    Attributes:
        _detector: YOLODetector 或 RKNNDetector 实例，用于推理
        _mapping: {class_id: sku_id} 的字典（从 model_mapping.yaml 加载）
    """

    def __init__(self, detector, mapping_path: str):
        """
        Args:
            detector: 已 load() 的 YOLODetector 或 RKNNDetector 实例
            mapping_path: model_mapping.yaml 路径
        """
        self._detector = detector
        self._mapping = self._load_mapping(mapping_path)
        logger.info(f"商品分类器已初始化，共 {len(self._mapping)} 个 SKU 类别")

    @staticmethod
    def _load_mapping(path: str) -> dict[int, str]:
        """从 model_mapping.yaml 加载 class_id -> sku_id 映射。

        YAML 格式：
            mapping:
              0: {sku_id: "cola_330ml", name: "可口可乐 330ml"}
              1: {sku_id: "sprite_330ml", name: "雪碧 330ml"}
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return {
            int(cid): info["sku_id"]
            for cid, info in data.get("mapping", {}).items()
        }

    def classify(self, frame: np.ndarray) -> tuple[str | None, float]:
        """对单帧图像做商品分类。

        流程：
        1. 调用检测器得到所有检测框
        2. 按置信度降序，取第一个框
        3. 将 class_id 映射为 sku_id

        Args:
            frame: BGR 图像 (H, W, 3)

        Returns:
            tuple:
                - sku_id: 识别到的 SKU 字符串，无法识别时为 None
                - confidence: 检测置信度 [0, 1]
        """
        detections = self._detector.detect(frame)

        if len(detections) == 0:
            logger.debug("YOLO 未检测到任何商品")
            return None, 0.0

        # 取置信度最高的检测框（detections 列: [x1,y1,x2,y2,conf,class_id]）
        best = detections[np.argmax(detections[:, 4])]
        class_id = int(best[5])
        confidence = float(best[4])

        sku_id = self._mapping.get(class_id)
        if sku_id is None:
            logger.warning(f"class_id={class_id} 不在 model_mapping.yaml 中")

        logger.info(f"视觉分类: class_id={class_id} → sku={sku_id}, conf={confidence:.2f}")
        return sku_id, confidence

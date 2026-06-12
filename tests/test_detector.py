# -*- coding: utf-8 -*-
"""YOLO 检测器单元测试。"""

import numpy as np
import pytest

from pipeline.detector.yolo_detector import YOLODetector


class TestYOLODetector:
    """测试 YOLO 检测器的工具方法（不需要 ONNX 模型）。"""

    def test_cxcywh_to_xyxy(self):
        """测试坐标格式转换 cxcywh -> xyxy。"""
        input_boxes = np.array([
            [100, 100, 50, 50],   # 中心 (100,100), 宽高 50x50 -> (75,75)-(125,125)
            [200, 300, 100, 80],  # 中心 (200,300), 宽高 100x80 -> (150,260)-(250,340)
        ])
        expected = np.array([
            [75, 75, 125, 125],
            [150, 260, 250, 340],
        ])
        result = YOLODetector._cxcywh_to_xyxy(input_boxes)
        np.testing.assert_array_almost_equal(result, expected)

    def test_nms_single_box(self):
        """测试 NMS 单框场景：应该保留。"""
        boxes = np.array([[0, 0, 100, 100]])
        scores = np.array([0.9])
        indices = YOLODetector._nms(boxes, scores, iou_threshold=0.5)
        assert len(indices) == 1
        assert indices[0] == 0

    def test_nms_overlapping_boxes(self):
        """测试 NMS 重叠框场景：抑制低分重叠框。"""
        boxes = np.array([
            [0, 0, 100, 100],    # 高分框 A
            [10, 10, 90, 90],    # 低分框 B，与 A 高度重叠 -> 应被抑制
            [200, 200, 300, 300],  # 框 C，不与 A/B 重叠 -> 应保留
        ])
        scores = np.array([0.9, 0.7, 0.8])
        indices = YOLODetector._nms(boxes, scores, iou_threshold=0.5)
        # A 和 C 应保留，B 被抑制
        assert len(indices) == 2
        assert 0 in indices
        assert 2 in indices

    def test_nms_no_overlap(self):
        """测试 NMS 无重叠场景：全部保留。"""
        boxes = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300],
            [400, 400, 500, 500],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        indices = YOLODetector._nms(boxes, scores, iou_threshold=0.5)
        assert len(indices) == 3

    def test_model_not_found(self):
        """测试模型文件不存在时抛出异常。"""
        with pytest.raises(FileNotFoundError):
            YOLODetector("non_existent_model.onnx")

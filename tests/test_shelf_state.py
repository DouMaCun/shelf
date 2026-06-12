# -*- coding: utf-8 -*-
"""货架状态管理单元测试。"""

import numpy as np
import pytest

from pipeline.shelf_state import ShelfState, ShelfConfig, SlotState


# 测试用的简配版货架配置
@pytest.fixture
def shelf_config() -> ShelfConfig:
    return ShelfConfig(
        shelf_id="test_shelf",
        camera_id="test_cam",
        width=1280,
        height=720,
        slots=[
            {"id": "A1", "roi": [0.0, 0.0, 0.5, 0.5], "sku_id": "item_A"},
            {"id": "A2", "roi": [0.5, 0.0, 1.0, 0.5], "sku_id": "item_B"},
            {"id": "B1", "roi": [0.0, 0.5, 0.5, 1.0], "sku_id": "item_C"},
            {"id": "B2", "roi": [0.5, 0.5, 1.0, 1.0], "sku_id": "item_D"},
        ],
    )


class TestShelfState:
    """测试货架状态管理器。"""

    def test_initialization(self, shelf_config):
        """测试初始化：所有槽位初始状态为 unoccupied。"""
        shelf = ShelfState(shelf_config, debounce_frames=3)
        assert len(shelf.slots) == 4
        for slot in shelf.slots.values():
            assert slot.occupied is False
            assert slot.matched_track_id == -1

    def test_track_to_slot_matching(self, shelf_config):
        """测试 track 与槽位的 IoU 匹配。

        A1 槽位像素: [0, 0, 640, 360], area = 230400
        track 框 [50, 50, 600, 300], area = 137500
        IoU = 137500/230400 = 0.597 > 0.3
        """
        shelf = ShelfState(shelf_config, min_iou_for_slot=0.3, debounce_frames=1)

        tracks = np.array([[50, 50, 600, 300, 1, 0, 0.95]], dtype=np.float32)

        occupancy = shelf.update(tracks, timestamp=1000.0)

        assert occupancy["A1"] is True
        assert occupancy["A2"] is False
        assert occupancy["B1"] is False
        assert occupancy["B2"] is False

    def test_multiple_tracks_different_slots(self, shelf_config):
        """测试多个 track 命中不同槽位。"""
        shelf = ShelfState(shelf_config, min_iou_for_slot=0.3, debounce_frames=1)

        # track_1 在 A1 [0,0,640,360]，track_2 在 B2 [640,360,1280,720]
        tracks = np.array([
            [50, 50, 600, 300, 1, 0, 0.95],          # A1
            [700, 400, 1200, 680, 2, 1, 0.91],        # B2
        ], dtype=np.float32)

        occupancy = shelf.update(tracks, timestamp=1000.0)

        assert occupancy["A1"] is True
        assert occupancy["A2"] is False
        assert occupancy["B1"] is False
        assert occupancy["B2"] is True

    def test_debounce(self, shelf_config):
        """测试去抖：只有连续 N 帧一致才改变状态。"""
        shelf = ShelfState(shelf_config, min_iou_for_slot=0.3, debounce_frames=3)

        track_a1 = np.array([[50, 50, 600, 300, 1, 0, 0.95]], dtype=np.float32)
        empty = np.empty((0, 7), dtype=np.float32)

        # 帧 1-2: 有商品（持续确认）
        occupancy = shelf.update(track_a1, timestamp=1.0)
        assert occupancy["A1"] is True   # 初始帧直接确认
        occupancy = shelf.update(track_a1, timestamp=2.0)
        assert occupancy["A1"] is True   # 持续确认

        # 帧 3: 突然没有了（去抖中，occupied 暂不变）
        occupancy = shelf.update(empty, timestamp=3.0)
        assert occupancy["A1"] is True   # 去抖未完成，仍为 True

        # 帧 4: 仍没有
        occupancy = shelf.update(empty, timestamp=4.0)
        assert occupancy["A1"] is True   # 去抖未完成

        # 帧 5: 仍没有 -> 去抖完成
        occupancy = shelf.update(empty, timestamp=5.0)
        assert occupancy["A1"] is False  # 已去抖，状态变更

    def test_get_snapshot(self, shelf_config):
        """测试货架状态快照。"""
        shelf = ShelfState(shelf_config, min_iou_for_slot=0.3, debounce_frames=1)
        tracks = np.array([[50, 50, 600, 300, 1, 0, 0.95]], dtype=np.float32)
        shelf.update(tracks)

        snapshot = shelf.get_snapshot()
        assert len(snapshot) == 4
        assert snapshot["A1"]["occupied"] is True
        assert snapshot["A1"]["sku_id"] == "item_A"

    def test_box_iou(self):
        """测试 IoU 计算。"""
        # 完全重叠
        iou = ShelfState._box_iou(
            np.array([0, 0, 100, 100]),
            (0, 0, 100, 100),
        )
        assert iou == 1.0

        # 完全不重叠
        iou = ShelfState._box_iou(
            np.array([0, 0, 100, 100]),
            (200, 200, 300, 300),
        )
        assert iou == 0.0

        # 部分重叠
        iou = ShelfState._box_iou(
            np.array([0, 0, 100, 100]),
            (50, 50, 150, 150),
        )
        assert 0.1 < iou < 0.2  # 交集 2500 / 并集 17500 = 0.143
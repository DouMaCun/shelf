# -*- coding: utf-8 -*-
"""事件引擎单元测试。"""

import numpy as np
import pytest

from pipeline.event_engine import EventEngine, ShelfEvent
from pipeline.shelf_state import ShelfState, ShelfConfig


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
        ],
    )


@pytest.fixture
def shelf_state(shelf_config) -> ShelfState:
    return ShelfState(shelf_config, debounce_frames=1)


class TestEventEngine:
    """测试事件判定引擎。"""

    def test_first_frame_no_event(self, shelf_state):
        """测试第一帧不产生事件（无前一帧比较）。"""
        engine = EventEngine()
        # 第一帧：所有槽位 occupied
        current_state = {"A1": True, "A2": True}
        events = engine.update(current_state, shelf_state)

        assert len(events) == 0

    def test_pick_event(self, shelf_state):
        """测试 pick 事件：occupied -> empty。"""
        engine = EventEngine()

        # 初始帧：两个槽位都 occupied
        engine.update({"A1": True, "A2": True}, shelf_state)

        # 下一帧：A1 变 empty
        events = engine.update({"A1": False, "A2": True}, shelf_state)

        assert len(events) == 1
        assert events[0].event_type == "pick"
        assert events[0].slot_id == "A1"
        assert events[0].sku_id == "item_A"
        assert events[0].track_id == -1  # pick 时无 track_id

    def test_place_event(self, shelf_state):
        """测试 place 事件：empty -> occupied。"""
        engine = EventEngine()

        # 初始帧：A1 empty
        engine.update({"A1": False, "A2": False}, shelf_state)

        # 下一帧：A1 变 occupied
        events = engine.update({"A1": True, "A2": False}, shelf_state)

        assert len(events) == 1
        assert events[0].event_type == "place"
        assert events[0].slot_id == "A1"
        assert events[0].sku_id == "item_A"

    def test_multiple_events_same_frame(self, shelf_state):
        """测试同一帧内多个事件。"""
        engine = EventEngine()

        # 初始帧
        engine.update({"A1": True, "A2": False}, shelf_state)

        # A1 pick, A2 place 同时发生
        events = engine.update({"A1": False, "A2": True}, shelf_state)

        assert len(events) == 2
        event_types = {e.event_type for e in events}
        assert "pick" in event_types
        assert "place" in event_types

    def test_callback(self, shelf_state):
        """测试事件回调。"""
        received = []

        def on_event(evt: ShelfEvent):
            received.append(evt)

        engine = EventEngine(on_event=on_event)
        engine.update({"A1": True}, shelf_state)
        engine.update({"A1": False}, shelf_state)

        assert len(received) == 1
        assert received[0].event_type == "pick"

    def test_state_updated_after_event(self, shelf_state):
        """测试事件产生后 previous_state 被正确更新。"""
        engine = EventEngine()

        engine.update({"A1": True}, shelf_state)
        engine.update({"A1": False}, shelf_state)  # 触发 pick

        # 再更新一次同样状态，不应产生事件
        events = engine.update({"A1": False}, shelf_state)
        assert len(events) == 0

    def test_reset(self, shelf_state):
        """测试 reset 清除历史。"""
        engine = EventEngine()
        engine.update({"A1": True}, shelf_state)
        engine.update({"A1": False}, shelf_state)  # 触发 pick

        assert len(engine.get_events()) == 1
        engine.reset()
        assert len(engine.get_events()) == 0

    def test_get_events_since(self, shelf_state):
        """测试按时间戳过滤事件。"""
        import time

        engine = EventEngine()
        engine.update({"A1": True}, shelf_state)

        t0 = time.time()
        engine.update({"A1": False}, shelf_state)  # 触发 pick

        # since=t0 后的事件应该包含 pick
        events = engine.get_events(since=t0 - 1)  # 比 t0 早一点
        assert len(events) == 1

        # since 太晚，没有事件
        events = engine.get_events(since=t0 + 100)
        assert len(events) == 0

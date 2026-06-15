# -*- coding: utf-8 -*-
"""
pick/place 事件判定引擎。

功能：
  基于帧间槽位占用状态的变化，判定「取走（pick）」和「放回（place）」两种事件。

核心逻辑：
  比较前后两帧的槽位占用状态字典：
  - 某槽位从 occupied=True -> occupied=False: 触发 pick 事件
  - 某槽位从 occupied=False -> occupied=True: 触发 place 事件

状态机示意：
  +-----------+          +----------+
  | OCCUPIED  | --pick-->|  EMPTY   |
  |(商品在架) |          |(商品离架)|
  +-----------+          +----------+
       ^                      |
       +-------place----------+

重要说明：
  去抖（debounce）已在 ShelfState 层完成，EventEngine 不需要再做去抖。
  EventEngine 收到的 current_state 已经是去抖后的稳定状态，
  所以这里的状态变化直接认定为有效事件。

事件数据结构：
  ShelfEvent: 包含事件 ID、时间戳、类型、SKU、槽位、置信度等完整信息。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from loguru import logger

from pipeline.shelf_state import ShelfState


# ============================================================================
# 事件数据结构
# ============================================================================

@dataclass
class ShelfEvent:
    """货架交互事件。

    一次 pick 或 place 行为对应一个 ShelfEvent 实例。

    Attributes:
        event_id: 事件唯一 ID（格式: "evt_" + 12 位随机 hex）
        timestamp: ISO 8601 格式的时间戳
        camera_id: 产生事件的摄像头 ID
        shelf_id: 产生事件的货架 ID
        event_type: 事件类型，"pick" 或 "place"
        sku_id: 涉及的商品 SKU ID
        slot_id: 涉及的槽位 ID
        track_id: 关联的 ByteTrack track_id（pick 时为 -1）
        confidence: 事件置信度（来自检测器）
        snapshot_path: 事件截图路径（仅当 output 层启用截图时）
        frame: 事件发生时的原始帧图像（numpy 数组，不序列化）
    """
    event_id: str
    timestamp: str
    camera_id: str
    shelf_id: str
    event_type: str               # "pick" | "place"
    sku_id: str
    slot_id: str
    track_id: int
    confidence: float
    snapshot_path: str = ""
    frame: np.ndarray | None = field(default=None, repr=False)


# ============================================================================
# 事件引擎
# ============================================================================

class EventEngine:
    """事件判定引擎。

    核心职责：
    1. 逐帧接收槽位占用状态，比较前后帧差异
    2. 发现状态变化时生成 ShelfEvent
    3. 通过回调函数通知外部（输出层）处理事件

    工作流程：
    Frame 1: previous_state = {A1: True, A2: True}  # 初始化
    Frame 2: current_state  = {A1: True, A2: False} # A2 变化
             -> 触发 pick 事件 (slot=A2)

    Attributes:
        camera_id: 摄像头 ID
        shelf_id: 货架 ID
        _on_event: 事件回调函数（可选），每次产生事件时调用
        _previous_state: 上一帧的槽位状态 {slot_id: bool}
        _events: 事件历史列表
    """

    def __init__(self, camera_id: str = "cam_01", shelf_id: str = "shelf_01",
                 on_event: Callable[[ShelfEvent], None] | None = None):
        """
        Args:
            camera_id: 摄像头标识
            shelf_id: 货架标识
            on_event: 事件回调函数，签名为 fn(event: ShelfEvent) -> None
        """
        self.camera_id = camera_id
        self.shelf_id = shelf_id
        self._on_event = on_event
        self._previous_state: dict[str, bool] = {}
        self._events: list[ShelfEvent] = []      # 事件历史（内存中保留）

    # ========================================================================
    # 核心方法
    # ========================================================================

    def update(self, current_state: dict[str, bool],
               shelf_state: ShelfState,
               frame: np.ndarray | None = None) -> list[ShelfEvent]:
        """比较前后帧槽位状态，生成事件。

        处理流程：
        1. 如果是第一帧（previous_state 为空），保存状态后返回空列表
        2. 遍历每个槽位，比较前后状态是否变化
        3. occupied=True -> False: pick
        4. occupied=False -> True: place
        5. 生成 ShelfEvent，加入历史列表
        6. 通过 on_event 回调通知外部
        7. 更新 previous_state

        Args:
            current_state: 当前帧的槽位状态 {slot_id: bool}
            shelf_state: ShelfState 对象（用于获取 SKU、track 等附加信息）
            frame: 当前帧图像（可选，用于保存事件截图）

        Returns:
            list[ShelfEvent]: 本帧产生的新事件列表
        """
        # 第一帧：只初始化状态，不产生事件
        if not self._previous_state:
            self._previous_state = dict(current_state)
            return []

        new_events: list[ShelfEvent] = []

        # 遍历每个槽位，检查状态是否变化
        for slot_id, occupied in current_state.items():
            prev_occupied = self._previous_state.get(slot_id, occupied)

            # 状态无变化，跳过
            if occupied == prev_occupied:
                continue

            # 获取槽位的 SKU 等详细信息
            slot = shelf_state.slots.get(slot_id)
            if slot is None:
                continue

            # 生成事件
            now = time.time()
            now_iso = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(now))

            # 判定事件类型
            if prev_occupied:
                # 之前有商品，现在没有了 -> pick
                event_type = "pick"
                track_id = -1  # pick 时没有关联 track
            else:
                # 之前没有商品，现在有了 -> place
                event_type = "place"
                track_id = slot.matched_track_id

            event = ShelfEvent(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                timestamp=now_iso,
                camera_id=self.camera_id,
                shelf_id=self.shelf_id,
                event_type=event_type,
                sku_id=slot.sku_id,
                slot_id=slot_id,
                track_id=track_id,
                confidence=slot.confidence,
                frame=frame.copy() if frame is not None else None,
            )
            new_events.append(event)
            self._events.append(event)

            # 日志输出
            logger.info(f"事件: {event_type.upper()} | slot={slot_id} | "
                        f"sku={slot.sku_id} | conf={slot.confidence:.2f}")

        # 更新状态缓存，供下一帧比较
        self._previous_state = dict(current_state)

        # 触发回调
        if self._on_event:
            for evt in new_events:
                self._on_event(evt)

        return new_events

    # ========================================================================
    # 查询接口
    # ========================================================================

    def get_events(self, since: float | None = None) -> list[ShelfEvent]:
        """查询历史事件。

        用于 REST API (/api/events) 的业务后端查询近期事件。

        Args:
            since: Unix 时间戳，只返回此时间之后的事件。None 表示返回全部。

        Returns:
            list[ShelfEvent]: 符合条件的事件列表
        """
        if since is None:
            return list(self._events)

        # 按时间戳过滤
        result = []
        for e in self._events:
            event_ts = time.mktime(
                time.strptime(e.timestamp, "%Y-%m-%dT%H:%M:%S"))
            if event_ts > since:
                result.append(e)
        return result

    def emit_pick(self, sku_id: str | None, confidence: float,
                  frame: np.ndarray | None = None) -> ShelfEvent:
        """直接发出一个 pick 事件（出口区识别模式使用）。

        与 update() 不同，此方法不做帧间状态对比，直接生成事件。
        适用于摄像头俯拍柜门、商品从出口区识别的场景。

        Args:
            sku_id: 识别到的 SKU，无法识别时传 None（事件仍会发出，sku 为 "unknown"）
            confidence: 识别置信度 [0, 1]
            frame: 识别时使用的帧图像（可选，用于截图存档）

        Returns:
            ShelfEvent: 生成的事件对象
        """
        now = time.time()
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))

        event = ShelfEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            timestamp=now_iso,
            camera_id=self.camera_id,
            shelf_id=self.shelf_id,
            event_type="pick",
            sku_id=sku_id or "unknown",
            slot_id="",        # 出口区模式不使用槽位
            track_id=-1,
            confidence=confidence,
            frame=frame.copy() if frame is not None else None,
        )
        self._events.append(event)
        logger.info(f"事件: PICK | sku={event.sku_id} | conf={confidence:.2f}")

        if self._on_event:
            self._on_event(event)

        return event

    def reset(self) -> None:
        """重置引擎状态和事件历史。

        用于数据源切换时清空旧的事件记录。
        """
        self._previous_state.clear()
        self._events.clear()

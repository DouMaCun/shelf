# -*- coding: utf-8 -*-
"""
货架槽位状态管理。

功能：
  维护货架上每个槽位的当前占用状态，是连接「目标检测/跟踪」和「事件判定」的中间层。

核心设计：槽位 RoI（Slot-based Region of Interest）
  货架被划分为若干个固定槽位，每个槽位对应一个预设的商品 SKU。
  每个槽位只追踪一个简单的二值状态：当前是否有商品占据（occupied: True/False）。

为什么用槽位而不是纯检测框：
  1. 鲁棒性更强：槽位位置固定，不受检测框的帧间抖动影响
  2. 问题简化：从「N 个移动目标的状态追踪」简化为「M 个固定区域的状态监控」
  3. 即插即用：货架布局确定后，槽位配置一次标注即可长期使用
  4. 抗遮挡：商品短暂被手遮挡时，只要检测框还能与槽位匹配上，状态就不会丢失

数据结构：
  SlotState: 单个槽位的运行时状态，包含：
    - occupied: 当前是否被商品占据
    - matched_track_id: 匹配到的跟踪 ID（用于跨帧关联）
    - confidence: 当前占据该槽位的检测置信度
    - _debounce_counter: 去抖计数器

去抖（debounce）机制：
  YOLO 检测存在帧间 flicker（同一目标在相邻帧的置信度波动），
  可能导致同一个槽位的 occupied 状态在 True/False 之间反复切换。
  去抖要求状态变化必须连续持续 N 帧才会被确认，从而过滤掉短暂的误检。

状态转换示例：
  帧 1:  YOLO 检测到槽位 A 有商品 -> raw_occupied=True  [debounce=1, pending=True]
  帧 2:  检测到槽位 A 有商品     -> raw_occupied=True  [debounce=2, pending=True]
  帧 3:  检测到槽位 A 有商品     -> raw_occupied=True  [debounce=3, pending=True]
  帧 4:  检测到槽位 A 有商品     -> raw_occupied=True  [debounce=4, pending=True]
  帧 5:  检测到槽位 A 有商品     -> raw_occupied=True  [debounce=5, 确认 occupied=True]
  帧 6:  检测到槽位 A 无商品     -> raw_occupied=False [debounce=1, pending=False]
  ...需要再连续 4 帧无商品才会确认 occupied=False...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml
from loguru import logger


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SlotState:
    """单个货架槽位的运行时状态。

    Attributes:
        slot_id: 槽位唯一标识（如 "A1", "B3"）
        sku_id: 该槽位对应的商品 SKU ID
        roi: 归一化坐标 [x1, y1, x2, y2]
        description: 槽位的文字描述（如 "可口可乐 330ml - 第一层左1"）
        occupied: 经过去抖确认后的当前占用状态
        matched_track_id: 当前匹配到的 ByteTrack track_id（-1 表示无匹配）
        confidence: 当前检测置信度 [0, 1]
        last_seen: 最后一次检测到商品的时间戳
        _debounce_counter: 去抖计数器（连续 N 帧状态保持一致）
        _pending_state: 待确认的候选状态
    """
    slot_id: str
    sku_id: str
    roi: list[float]              # 归一化坐标 [x1, y1, x2, y2]
    description: str = ""

    # ---- 运行时状态 ----
    occupied: bool = False        # 当前是否被商品占据（去抖后的确认值）
    matched_track_id: int = -1    # 当前匹配的跟踪 ID，-1 表示无匹配
    confidence: float = 0.0       # 当前检测置信度
    last_seen: float = 0.0        # 最后检测到的时间戳（Unix timestamp）

    # ---- 去抖相关 ----
    _debounce_counter: int = 0    # 连续帧状态不变的计数
    _pending_state: bool | None = None  # 待确认的状态（None 表示刚初始化）


@dataclass
class ShelfConfig:
    """货架静态配置（从 YAML 文件加载）。

    Attributes:
        shelf_id: 货架唯一标识
        camera_id: 关联的摄像头 ID
        width: 摄像头画面宽度（用于归一化坐标转像素）
        height: 摄像头画面高度
        slots: 槽位配置列表，每个元素是 {id, roi, sku_id, description} 字典
    """
    shelf_id: str
    camera_id: str
    width: int
    height: int
    slots: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> 'ShelfConfig':
        """从 YAML 配置文件加载货架布局。

        Args:
            path: shelf_layout.yaml 的路径

        Returns:
            ShelfConfig: 货架配置对象

        Example:
            >>> config = ShelfConfig.from_yaml("config/shelf_layout.yaml")
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(
            shelf_id=data['shelf_id'],
            camera_id=data['camera_id'],
            width=data['width'],
            height=data['height'],
            slots=data.get('slots', []),
        )


# ============================================================================
# 货架状态管理器
# ============================================================================

class ShelfState:
    """货架状态管理器。

    核心职责：
    1. 维护所有槽位的当前状态
    2. 每帧用跟踪结果更新槽位状态（通过 IoU 匹配 track 和槽位）
    3. 对状态变化做去抖处理
    4. 为事件判定引擎提供当前帧的状态快照

    使用方式：
        shelf = ShelfState(config, min_iou_for_slot=0.3, debounce_frames=5)
        while True:
            tracks = tracker.update(detections)
            slot_occupancy = shelf.update(tracks, timestamp)
            # slot_occupancy 是去抖后的稳定状态
    """

    def __init__(self, config: ShelfConfig, min_iou_for_slot: float = 0.3,
                 debounce_frames: int = 5):
        """
        Args:
            config: 货架配置（槽位布局、SKU 映射）
            min_iou_for_slot: track 检测框与槽位匹配的最小 IoU 阈值，默认 0.3
            debounce_frames: 去抖帧数，需要连续 N 帧状态一致才确认，默认 5
        """
        self.config = config
        self.min_iou = min_iou_for_slot
        self.debounce_frames = debounce_frames
        self.slots: dict[str, SlotState] = {}

        # 根据配置初始化所有槽位状态
        for slot_cfg in config.slots:
            self.slots[slot_cfg['id']] = SlotState(
                slot_id=slot_cfg['id'],
                sku_id=slot_cfg['sku_id'],
                roi=slot_cfg['roi'],
                description=slot_cfg.get('description', ''),
            )
        logger.info(f"货架状态已初始化: shelf={config.shelf_id}, 共 {len(self.slots)} 个槽位")

    # ========================================================================
    # 状态更新（核心方法）
    # ========================================================================

    def update(self, tracks: np.ndarray, timestamp: float | None = None) -> dict[str, bool]:
        """用当前帧的跟踪结果更新所有槽位状态。

        更新流程：
        1. 将当前帧的所有 track 检测框与每个槽位 RoI 计算 IoU
        2. 对于每个槽位，找到与其 IoU 最高且 >= min_iou 的 track
        3. 如果多个 track 竞争同一个槽位，保留置信度最高的
        4. 执行去抖：检查状态是否连续 N 帧一致
        5. 返回去抖后的槽位占用字典

        Args:
            tracks: ByteTracker 输出的活跃 track 数组，shape (M, 7)，
                    列含义: [x1, y1, x2, y2, track_id, class_id, score]
            timestamp: 当前帧的时间戳，用于记录 last_seen

        Returns:
            dict[str, bool]: slot_id -> is_occupied，去抖后的稳定占用状态
        """
        if timestamp is None:
            timestamp = time.time()

        w, h = self.config.width, self.config.height

        # ---- 第一步：匹配 track 与槽位 ----
        # current_slot_match: {slot_id: (track_id, score)}
        current_slot_match: dict[str, tuple[int, float]] = {}

        for i in range(len(tracks)):
            track_box = tracks[i, :4]   # 检测框 [x1, y1, x2, y2]
            track_id = int(tracks[i, 4])
            score = tracks[i, 6]

            # 在槽位中找与其 IoU 最高且 >= min_iou 的槽位
            best_slot = None
            best_iou = 0.0

            for slot in self.slots.values():
                # 归一化坐标转像素坐标
                sx1 = slot.roi[0] * w
                sy1 = slot.roi[1] * h
                sx2 = slot.roi[2] * w
                sy2 = slot.roi[3] * h
                iou = self._box_iou(track_box, (sx1, sy1, sx2, sy2))
                if iou > best_iou and iou >= self.min_iou:
                    best_iou = iou
                    best_slot = slot.slot_id

            if best_slot is not None:
                # 如果此槽位已被另一个 track 占据，保留置信度更高的
                if best_slot in current_slot_match:
                    if score > current_slot_match[best_slot][1]:
                        current_slot_match[best_slot] = (track_id, score)
                else:
                    current_slot_match[best_slot] = (track_id, score)

        # ---- 第二步：应用去抖，更新状态 ----
        slot_occupancy = {}
        for slot_id, slot in self.slots.items():
            # 判断当前帧该槽位是否有商品（是否匹配到了 track）
            is_occupied = slot_id in current_slot_match

            if is_occupied:
                slot.matched_track_id = current_slot_match[slot_id][0]
                slot.confidence = current_slot_match[slot_id][1]
                slot.last_seen = timestamp
            else:
                slot.matched_track_id = -1

            # 去抖：连续 N 帧一致才真正改变 occupied
            determined = self._apply_debounce(slot, is_occupied)
            slot_occupancy[slot_id] = determined

        return slot_occupancy

    # ========================================================================
    # 去抖逻辑
    # ========================================================================

    def _apply_debounce(self, slot: SlotState, raw_occupied: bool) -> bool:
        """去抖处理：只有连续 N 帧状态一致，才确认状态变更。

        实现一个简单的迟滞比较器（hysteresis comparator）：
        - 如果当前帧的原始状态与「待确认状态」一致，计数器 +1
        - 如果不一致，重置计数器，并将待确认状态更新为当前帧的原始状态
        - 当计数器达到 debounce_frames，正式确认状态变更

        Args:
            slot: 槽位状态对象
            raw_occupied: 当前帧的原始 occupied 状态（去抖前）

        Returns:
            bool: 去抖后的 occupied 状态
        """
        if slot._pending_state is None:
            # 首次初始化：直接确认，无需去抖
            slot._pending_state = raw_occupied
            slot._debounce_counter = 1
            slot.occupied = raw_occupied
            return raw_occupied

        if raw_occupied == slot._pending_state:
            # 状态一致，计数器递增
            slot._debounce_counter += 1
        else:
            # 状态变化，重置去抖过程
            slot._pending_state = raw_occupied
            slot._debounce_counter = 1

        # 计数器达到阈值，确认状态变更
        if slot._debounce_counter >= self.debounce_frames:
            slot.occupied = slot._pending_state

        return slot.occupied

    # ========================================================================
    # 查询接口
    # ========================================================================

    def get_snapshot(self) -> dict[str, dict]:
        """获取当前所有槽位的状态快照。

        用于 REST API (/api/shelf/state) 返回给前端或业务后端。

        Returns:
            dict: {
                "A1": {
                    "slot_id": "A1",
                    "sku_id": "cola_330ml",
                    "occupied": true,
                    "confidence": 0.95,
                    "last_seen": 1718123456.789
                },
                ...
            }
        """
        snapshot = {}
        for slot_id, slot in self.slots.items():
            snapshot[slot_id] = {
                "slot_id": slot.slot_id,
                "sku_id": slot.sku_id,
                "occupied": slot.occupied,
                "confidence": slot.confidence,
                "last_seen": slot.last_seen,
            }
        return snapshot

    def get_previous_state(self) -> dict[str, bool]:
        """获取所有槽位的当前状态（供事件引擎做帧间比较）。"""
        return {sid: s.occupied for sid, s in self.slots.items()}

    # ========================================================================
    # 工具方法
    # ========================================================================

    @staticmethod
    def _box_iou(box1: np.ndarray, box2: tuple[float, ...]) -> float:
        """计算两个矩形框的 IoU（交并比）。

        Args:
            box1: 检测框 [x1, y1, x2, y2]
            box2: 槽位 Roi [x1, y1, x2, y2]

        Returns:
            float: IoU 值 [0, 1]
        """
        # 交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)

        # 并集区域
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / max(union, 1e-6)

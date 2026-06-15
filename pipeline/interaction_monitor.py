# -*- coding: utf-8 -*-
"""
交互监控器——替代原 ShelfState 的核心状态机。

适用场景：摄像头装在柜门顶部向下俯拍，商品随机摆放。
任何被取走的商品都必须经过柜门开口（摄像头正下方），取出时商品会举起穿过视野。

状态机：
  IDLE ──手臂进入──► ACTIVE ──手臂离开──► CLASSIFYING ──识别完成──► COOLDOWN ──► IDLE
           缓冲帧队列积累中           选最清晰帧执行识别

各状态说明：
  IDLE        : 低频采样，等待有人伸手进柜
  ACTIVE      : 检测到手臂，切高频采样，持续缓冲帧
  CLASSIFYING : 手臂离开后，从缓冲帧中选最清晰帧执行商品识别（外部驱动）
  COOLDOWN    : 事件已发出，静默 N 秒防止重复触发，然后回到 IDLE

"最清晰帧"定义：用 Laplacian 算子的方差衡量图像模糊程度，方差越大越清晰。
  货架场景中，商品被举起穿过镜头视野时通常最清晰（最接近摄像头）。
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
from loguru import logger


class InteractionState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    CLASSIFYING = "classifying"
    COOLDOWN = "cooldown"


class InteractionMonitor:
    """交互状态机，管理从"等待"到"识别商品"的完整流程。

    Attributes:
        cooldown_seconds: 事件发出后的静默时长（秒）
        frame_buffer_size: ACTIVE 阶段最多缓冲多少帧
        _state: 当前状态
        _frame_buffer: 帧缓冲队列
        _cooldown_until: 冷却结束的时间戳
        _active_since: 进入 ACTIVE 的时间戳（用于调试）
    """

    def __init__(self, cooldown_seconds: float = 3.0, frame_buffer_size: int = 30):
        """
        Args:
            cooldown_seconds: 识别完成后静默多少秒，防止重复触发
            frame_buffer_size: ACTIVE 状态下最多缓冲多少帧
        """
        self.cooldown_seconds = cooldown_seconds
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=frame_buffer_size)
        self._state = InteractionState.IDLE
        self._cooldown_until = 0.0
        self._active_since = 0.0

    # ========================================================================
    # 核心更新（每帧调用）
    # ========================================================================

    def update(self, frame: np.ndarray, has_person: bool) -> InteractionState:
        """根据当前帧的手臂检测结果推进状态机。

        调用方每帧调用此方法，传入当前帧和"是否检测到手臂"的布尔值。
        状态机自动推进；当状态变为 CLASSIFYING 时，调用方应调用
        get_best_frame() 获取最佳帧并执行识别，识别完成后调用
        complete_classification() 进入冷却。

        Args:
            frame: 当前帧 (BGR ndarray)
            has_person: 当前帧是否检测到手臂/人体

        Returns:
            InteractionState: 更新后的状态（供调用方判断是否触发识别）
        """
        now = time.time()
        prev_state = self._state

        if self._state == InteractionState.IDLE:
            if has_person:
                self._state = InteractionState.ACTIVE
                self._active_since = now
                self._frame_buffer.clear()
                logger.info("IDLE → ACTIVE：检测到手臂进入视野")

        elif self._state == InteractionState.ACTIVE:
            self._frame_buffer.append(frame.copy())
            if not has_person:
                self._state = InteractionState.CLASSIFYING
                logger.info(
                    f"ACTIVE → CLASSIFYING：手臂离开，缓冲 {len(self._frame_buffer)} 帧，"
                    f"交互时长 {now - self._active_since:.1f}s"
                )

        elif self._state == InteractionState.COOLDOWN:
            if now >= self._cooldown_until:
                self._state = InteractionState.IDLE
                logger.info("COOLDOWN → IDLE")

        # CLASSIFYING 状态下不自动推进，等外部调用 complete_classification()

        return self._state

    def complete_classification(self) -> None:
        """识别流程完成，进入冷却期。

        调用方在识别出 SKU（或识别失败）后调用此方法。
        状态机进入 COOLDOWN，N 秒后自动回到 IDLE。
        """
        self._state = InteractionState.COOLDOWN
        self._cooldown_until = time.time() + self.cooldown_seconds
        self._frame_buffer.clear()
        logger.info(f"CLASSIFYING → COOLDOWN（{self.cooldown_seconds}s）")

    # ========================================================================
    # 最佳帧选择
    # ========================================================================

    def get_best_frame(self) -> np.ndarray | None:
        """从缓冲帧中选出最清晰的一帧，用于商品识别。

        使用 Laplacian 方差衡量图像清晰度：
        - 方差大 → 边缘丰富 → 清晰
        - 方差小 → 模糊（运动模糊或离焦）

        商品被举起穿过镜头时通常最接近摄像头，此时图像最清晰，
        因此最清晰帧往往就是商品在出口区的最佳识别时刻。

        Returns:
            最清晰帧的 BGR ndarray，缓冲区为空时返回 None
        """
        if not self._frame_buffer:
            return None

        best_frame = None
        best_score = -1.0

        for f in self._frame_buffer:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if score > best_score:
                best_score = score
                best_frame = f

        logger.debug(f"最佳帧清晰度得分: {best_score:.1f}")
        return best_frame

    # ========================================================================
    # 查询接口
    # ========================================================================

    @property
    def state(self) -> InteractionState:
        """当前状态。"""
        return self._state

    @property
    def buffer_size(self) -> int:
        """当前缓冲帧数量。"""
        return len(self._frame_buffer)

    def reset(self) -> None:
        """重置到 IDLE 状态，清空缓冲。用于数据源切换或异常恢复。"""
        self._state = InteractionState.IDLE
        self._frame_buffer.clear()
        self._cooldown_until = 0.0
        logger.info("InteractionMonitor 已重置")

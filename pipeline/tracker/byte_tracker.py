# -*- coding: utf-8 -*-
"""
ByteTrack 多目标跟踪器。

功能：
  将逐帧的检测框关联为稳定的目标轨迹（track）。
  当同一商品在多帧中出现时，ByteTrack 会为它分配一个不变的 track_id，
  从而可以追踪该商品在画面中的持续存在或消失。

为什么用 ByteTrack 而不是 DeepSORT：
  1. ByteTrack 不需要额外的 ReID 特征提取网络，推理更快
  2. 货架场景中目标相对静态，移动主要发生在 pick/place 瞬间
  3. ByteTrack 的低分检测框二次匹配策略，能有效处理遮挡和截断

核心算法：
  ByteTrack 与 SORT 类似，但关键区别在于：
  - 高分检测框：直接与现有 track 做第一次 IoU 匹配
  - 低分检测框：利用第一次匹配后剩余的未匹配 track，做第二次 IoU 匹配
  这个「二次匹配」机制能有效恢复被短暂遮挡的目标，减少 ID-Switch。

数据结构：
  Track: 单个跟踪目标的状态
    - track_id: 唯一标识
    - velocity: 简单速度模型，用于预测下一帧位置
    - hits: 连续匹配帧数（用于确认 track 是否稳定）
    - time_since_update: 失配帧数（超过 track_buffer 则删除）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


# ============================================================================
# Track 数据结构
# ============================================================================

@dataclass
class Track:
    """单个跟踪目标的状态快照。

    维护目标在当前帧以及历史帧中的所有信息，用于运动预测和生命周期管理。

    Attributes:
        track_id: 目标唯一 ID，由 ByteTracker 自动分配
        class_id: 目标类别 ID（对应 YOLO class_id）
        x1, y1, x2, y2: 当前检测框坐标（像素）
        score: 当前帧的检测置信度
        velocity: 4 维速度向量 [vx1, vy1, vx2, vy2]，用于预测下一帧位置
        age: 目标从创建到现在的总帧数
        time_since_update: 最近一次成功匹配后经过的帧数（用于判断是否删除）
        hits: 连续成功匹配的帧数（>=3 时视为「已确认」）
        history: 最近 30 帧的检测框历史（可用于轨迹平滑）
    """
    track_id: int
    class_id: int
    # 当前边界框坐标
    x1: float; y1: float; x2: float; y2: float
    score: float
    # 运动模型：用最近一次匹配的位移作为速度估计
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # 生命周期状态
    age: int = 0                     # 总存活帧数
    time_since_update: int = 0       # 失配帧数（用于超时删除）
    hits: int = 0                    # 连续匹配帧数（用于确认稳定性）
    # 历史轨迹（保留最近 30 帧）
    history: list[np.ndarray] = field(default_factory=list)

    def predict(self) -> np.ndarray:
        """用匀速运动模型预测当前帧的边界框位置。

        预测公式: box_t = box_{t-1} + velocity
        这是一个简化的 Kalman 替代方案，假设目标在相邻帧间匀速运动。

        Returns:
            np.ndarray: 预测的边界框 [x1, y1, x2, y2]
        """
        box = np.array([self.x1, self.y1, self.x2, self.y2])
        return box + self.velocity

    def update(self, box: np.ndarray, score: float) -> None:
        """用新的检测框更新跟踪状态。

        执行以下更新：
        1. 计算速度（新位置 - 旧位置）
        2. 更新边界框和置信度
        3. 增加年龄和命中计数
        4. 重置失配计数器
        5. 将当前框加入历史记录

        Args:
            box: 匹配到的检测框 [x1, y1, x2, y2]
            score: 检测置信度
        """
        self.velocity = box - np.array([self.x1, self.y1, self.x2, self.y2])
        self.x1, self.y1, self.x2, self.y2 = box
        self.score = score
        self.age += 1
        self.time_since_update = 0             # 重置失配计数
        self.hits += 1                         # 命中 +1
        self.history.append(box)
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self) -> None:
        """标记本帧未匹配到检测框。

        增加年龄和失配计数，如果失配过久将被 ByteTracker 删除。
        """
        self.age += 1
        self.time_since_update += 1

    def is_confirmed(self) -> bool:
        """判断 track 是否已确认（稳定）。

        ByteTrack 论文中建议 hits >= 3 才视为有效 track，
        避免将短暂的误检作为真实目标输出。

        Returns:
            bool: hits >= 3 时为 True
        """
        return self.hits >= 3

    def to_array(self) -> np.ndarray:
        """将 track 转为输出数组。

        输出格式提供了完整的 track 信息，可供上层模块（货架状态、事件判定）直接使用。

        Returns:
            np.ndarray: [x1, y1, x2, y2, track_id, class_id, score], shape (7,)
        """
        return np.array([self.x1, self.y1, self.x2, self.y2,
                         self.track_id, self.class_id, self.score], dtype=np.float32)


# ============================================================================
# ByteTracker 主类
# ============================================================================

class ByteTracker:
    """ByteTrack 多目标跟踪器。

    实现 ByteTrack 论文中的核心匹配策略：高分-低分二次匹配。

    每一帧的处理流程：
    1. 用速度模型预测所有 track 的当前位置
    2. 将检测框按置信度分为「高分框」和「低分框」
    3. 第一次匹配：高分框与所有 track 做 IoU 匹配
    4. 第二次匹配：第一次匹配中未匹配的 track 与低分框做 IoU 匹配
    5. 为仍未匹配的高分框创建新 track
    6. 删除超时未更新的 track

    Attributes:
        track_thresh: 高分/低分检测框的分界阈值，默认 0.5
        match_thresh: 匹配的 IoU 阈值，默认 0.8
        track_buffer: track 失配后保持的帧数，默认 30
        frame_rate: 视频帧率，用于运动模型
    """

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8,
                 track_buffer: int = 30, frame_rate: int = 5):
        """
        Args:
            track_thresh: 高分检测阈值，>= 此值的检测框参与第一次匹配
            match_thresh: IoU 匹配阈值
            track_buffer: 失配帧缓冲，track 在 time_since_update > track_buffer 时被删除
            frame_rate: 帧率，用于运动模型的参数调优
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate

        # track_id 自增计数器
        self._next_id = 0
        # 当前所有活跃的 track 列表
        self._tracks: list[Track] = []

    # ========================================================================
    # 公共接口
    # ========================================================================

    def update(self, detections: np.ndarray) -> np.ndarray:
        """核心方法：用新一帧的检测结果更新跟踪状态。

        完整的帧间跟踪流程（详见类文档字符串）。

        Args:
            detections: 检测框数组，shape (N, 6)，列含义：
                [0:4] 边界框 [x1, y1, x2, y2]
                [4]   置信度
                [5]   类别 ID

        Returns:
            np.ndarray: 所有已确认的活跃 track，shape (M, 7)，列含义：
                [0:4] 边界框 [x1, y1, x2, y2]
                [4]   track_id
                [5]   class_id
                [6]   置信度
        """
        # 确保 detections 是 numpy 数组
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 6), dtype=np.float32)

        # ---- Step 1: 按置信度分高低分 ----
        high_mask = detections[:, 4] >= self.track_thresh
        low_mask = ~high_mask & (detections[:, 4] >= 0.1)  # 极低置信度（<0.1）直接丢弃
        high_dets = detections[high_mask]
        low_dets = detections[low_mask]

        # ---- Step 2: 对所有 track 做速度模型预测，并标记本帧未匹配 ----
        for track in self._tracks:
            track.mark_missed()

        # ---- Step 3: 第一次匹配（高分框 vs 所有 track） ----
        # 返回三类结果：匹配对、未匹配的 track、未匹配的高分框
        matched, unmatched_tracks, unmatched_high = self._match(
            self._tracks, high_dets, self.match_thresh)

        # ---- Step 4: 第二次匹配（剩余未匹配 track vs 低分框） ----
        # 用更宽松的 IoU 阈值（0.5）处理低分框
        matched2, unmatched_tracks2, _ = self._match(
            unmatched_tracks, low_dets, 0.5)

        # 合并两次匹配的结果
        all_matched = matched + matched2
        unmatched_tracks = unmatched_tracks2

        # ---- Step 5: 更新已匹配的 track ----
        for track, det_box, det_score in all_matched:
            track.update(det_box, det_score)

        # ---- Step 6: 为未匹配的高分框创建新 track ----
        for det in unmatched_high:
            self._tracks.append(Track(
                track_id=self._next_id,
                class_id=int(det[5]),
                x1=det[0], y1=det[1], x2=det[2], y2=det[3],
                score=det[4],
            ))
            self._next_id += 1

        # ---- Step 7: 清理超时的 track ----
        # time_since_update > track_buffer 表示该目标已消失过久，删除
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.track_buffer]

        # ---- 返回所有已确认的活跃 track ----
        return self._get_active_tracks()

    # ========================================================================
    # 匹配算法
    # ========================================================================

    def _match(self, tracks: list[Track], detections: np.ndarray,
               match_thresh: float) -> tuple[list, list[Track], np.ndarray]:
        """基于 IoU 的贪心匹配算法。

        这是一个简化的匈牙利匹配：
        1. 计算所有 (track, detection) 对的 IoU 矩阵
        2. 按 IoU 从高到低排序所有可能的匹配对
        3. 贪心地选择 IoU 最高的未匹配对

        Args:
            tracks: 待匹配的 track 列表
            detections: 待匹配的检测框数组
            match_thresh: IoU 阈值

        Returns:
            tuple:
                - matched: [(track, box, score), ...] 成功匹配的对
                - unmatched_tracks: 未匹配的 track 列表
                - unmatched_dets: 未匹配的检测框数组
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], tracks, detections

        # 计算 IoU 矩阵: (n_tracks, n_dets)
        iou_matrix = self._compute_iou_matrix(tracks, detections)

        # 收集所有 IoU >= 阈值的配对，按 IoU 降序排列
        matched = []
        used_det: set[int] = set()   # 已匹配的检测框索引
        used_trk: set[int] = set()   # 已匹配的 track 索引

        pairs: list[tuple[float, int, int]] = []  # [(iou, track_idx, det_idx), ...]
        for t_idx, row in enumerate(iou_matrix):
            for d_idx, iou in enumerate(row):
                if iou >= match_thresh:
                    pairs.append((iou, t_idx, d_idx))
        pairs.sort(key=lambda x: -x[0])  # IoU 降序

        # 贪心匹配：优先满足 IoU 最高的配对
        for iou, t_idx, d_idx in pairs:
            if t_idx not in used_trk and d_idx not in used_det:
                track = tracks[t_idx]
                det = detections[d_idx]
                matched.append((track, det[:4], det[4]))  # (track, box, score)
                used_trk.add(t_idx)
                used_det.add(d_idx)

        # 收集未匹配的 track 和检测框
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in used_trk]
        unmatched_dets = detections[
            [i for i in range(len(detections)) if i not in used_det]
        ]

        return matched, unmatched_tracks, unmatched_dets

    # ========================================================================
    # IoU 计算
    # ========================================================================

    def _compute_iou_matrix(self, tracks: list[Track],
                            detections: np.ndarray) -> np.ndarray:
        """计算所有 track 的预测框与所有检测框之间的 IoU 矩阵。

        注意：
        - 对于 track，使用 predict() 预测的位置，而非当前位置
        - 预测位置 = 当前位置 + 速度向量（匀速运动假设）

        Args:
            tracks: track 列表
            detections: 检测框数组 (M, 6)

        Returns:
            np.ndarray: IoU 矩阵，shape (n_tracks, n_dets)
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            # 用速度模型预测 track 在当前帧的位置
            pred = track.predict()
            tx1, ty1, tx2, ty2 = pred

            for j, det in enumerate(detections):
                dx1, dy1, dx2, dy2 = det[:4]

                # 计算交集面积
                xx1 = max(tx1, dx1)
                yy1 = max(ty1, dy1)
                xx2 = min(tx2, dx2)
                yy2 = min(ty2, dy2)
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

                # 计算并集面积
                area_t = (tx2 - tx1) * (ty2 - ty1)
                area_d = (dx2 - dx1) * (dy2 - dy1)
                union = area_t + area_d - inter

                # IoU = 交集 / 并集
                iou_matrix[i, j] = inter / max(union, 1e-6)

        return iou_matrix

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _get_active_tracks(self) -> np.ndarray:
        """获取所有已确认的活跃 track，转为数组输出。

        仅返回 hits >= 3 的 track（已确认），过滤掉刚创建还不稳定的 track。

        Returns:
            np.ndarray: shape (M, 7), 每行 [x1, y1, x2, y2, track_id, class_id, score]
        """
        active = [t.to_array() for t in self._tracks if t.is_confirmed()]
        if not active:
            return np.empty((0, 7), dtype=np.float32)
        return np.stack(active)

    def reset(self) -> None:
        """重置所有跟踪状态。

        通常用于数据源切换（如换了一个视频文件）时，清空旧的 track 信息。
        """
        self._tracks.clear()
        self._next_id = 0

# -*- coding: utf-8 -*-
"""ByteTrack 跟踪器单元测试。"""

import numpy as np
import pytest

from pipeline.tracker.byte_tracker import ByteTracker, Track


class TestTrack:
    """测试 Track 数据结构的各项功能。"""

    def test_track_initialization(self):
        """测试 Track 初始状态。"""
        track = Track(track_id=1, class_id=3, x1=10, y1=20, x2=110, y2=120, score=0.95)
        assert track.track_id == 1
        assert track.class_id == 3
        assert not track.is_confirmed()  # hits=0, 未确认
        assert track.hits == 0

    def test_track_update_and_confirm(self):
        """测试 Track 更新和确认流程。"""
        track = Track(track_id=1, class_id=3, x1=10, y1=20, x2=110, y2=120, score=0.95)
        # 连续匹配 3 次
        track.update(np.array([15, 25, 115, 125]), 0.92)
        track.update(np.array([16, 26, 116, 126]), 0.93)
        assert not track.is_confirmed()
        track.update(np.array([17, 27, 117, 127]), 0.94)
        assert track.is_confirmed()  # hits=3

    def test_track_velocity(self):
        """测试 Track 速度模型。"""
        track = Track(track_id=1, class_id=3, x1=0, y1=0, x2=100, y2=100, score=0.95)
        # 更新到新位置，速度 = 新位置 - 旧位置
        track.update(np.array([10, 5, 110, 105]), 0.9)
        np.testing.assert_array_equal(track.velocity, np.array([10, 5, 10, 5]))

    def test_track_predict(self):
        """测试 Track 位置预测（匀速模型）。"""
        track = Track(track_id=1, class_id=3, x1=0, y1=0, x2=100, y2=100, score=0.95)
        track.update(np.array([10, 5, 110, 105]), 0.9)
        # predict 应返回当前位置 + 速度
        predicted = track.predict()
        expected = np.array([10, 5, 110, 105]) + np.array([10, 5, 10, 5])
        np.testing.assert_array_equal(predicted, expected)

    def test_track_mark_missed(self):
        """测试未匹配时的状态更新。"""
        track = Track(track_id=1, class_id=3, x1=10, y1=20, x2=110, y2=120, score=0.95)
        track.update(np.array([10, 20, 110, 120]), 0.9)
        assert track.time_since_update == 0
        track.mark_missed()
        assert track.time_since_update == 1
        assert track.age == 2

    def test_track_to_array(self):
        """测试 Track 序列化为数组。"""
        track = Track(track_id=5, class_id=2, x1=100, y1=200, x2=300, y2=400, score=0.88)
        arr = track.to_array()
        assert len(arr) == 7
        assert arr[0] == 100   # x1
        assert arr[3] == 400   # y2
        assert arr[4] == 5     # track_id
        assert arr[5] == 2     # class_id
        assert arr[6] == 0.88  # score


class TestByteTracker:
    """测试 ByteTracker 跟踪器行为。"""

    def test_empty_detections(self):
        """测试空检测框输入。"""
        tracker = ByteTracker()
        result = tracker.update(np.empty((0, 6), dtype=np.float32))
        assert len(result) == 0

    def test_single_detection_multiple_frames(self):
        """测试单目标持续跟踪。"""
        tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
        det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        # 连续 3 帧才能确认
        for _ in range(3):
            result = tracker.update(det)
        # 第 3 帧后应该有 1 个已确认的 track
        result = tracker.update(det)
        assert len(result) == 1
        assert result[0, 4] >= 0  # 有 track_id
        assert result[0, 5] == 0  # class_id 保持

    def test_track_expiry(self):
        """测试 track 超时删除。"""
        tracker = ByteTracker(track_thresh=0.5, track_buffer=2)
        det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        # 先让 track 确认
        for _ in range(4):
            tracker.update(det)
        assert len(tracker.update(det)) == 1
        # 连续 3 帧无检测，应该被删除
        tracker.update(np.empty((0, 6)))
        tracker.update(np.empty((0, 6)))
        result = tracker.update(np.empty((0, 6)))
        assert len(result) == 0

    def test_reset(self):
        """测试 reset 清除所有 track。"""
        tracker = ByteTracker()
        det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        for _ in range(4):
            tracker.update(det)
        assert len(tracker.update(det)) == 1
        tracker.reset()
        assert len(tracker.update(det)) == 0  # 重置后无 track

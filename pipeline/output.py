# -*- coding: utf-8 -*-
"""
事件输出层。

功能：
  接收 EventEngine 产生的事件，并通过多种通道输出：
  1. WebSocket 推送：实时推送到所有连接的 WebSocket 客户端
  2. 截图保存：将事件发生时的帧图像保存为 JPEG 文件
  3. SQLite 持久化：将事件信息写入本地 SQLite 数据库

职责：
  - 与业务后端对接的事件消费者（通过 WebSocket）
  - 本地数据留存（用于事后审计、离线分析和模型迭代）
  - 截图存档（用于人工复核事件准确性）

WebSocket 并发安全：
  使用 threading.Lock 保护 _ws_clients 集合的并发访问，
  确保同时注册/注销/广播不会导致集合损坏。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from pipeline.event_engine import ShelfEvent


class EventOutput:
    """事件输出管理器。

    负责事件的持久化、截图保存和实时推送。

    Attributes:
        save_snapshots: 是否保存事件截图到磁盘
        snapshot_dir: 截图保存目录
        db_path: SQLite 数据库文件路径（空字符串表示不启用）
        _ws_clients: 同步模式下的 WebSocket 客户端集合
        _lock: 保护 _ws_clients 的线程锁
        _db_conn: SQLite 数据库连接（启用时创建）
    """

    def __init__(self, save_snapshots: bool = False, snapshot_dir: str = "data/snapshots",
                 db_path: str = ""):
        """
        Args:
            save_snapshots: 是否保存事件截图，默认 False
            snapshot_dir: 截图保存目录，默认 "data/snapshots"
            db_path: SQLite 数据库路径，空字符串表示不启用 SQLite 持久化
        """
        self.save_snapshots = save_snapshots
        self.snapshot_dir = Path(snapshot_dir)
        self.db_path = db_path
        self._ws_clients: set = set()
        self._lock = threading.Lock()
        self._db_conn: sqlite3.Connection | None = None

        # 创建截图目录
        if save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 SQLite 数据库
        if db_path:
            self._init_db()

    # ========================================================================
    # 数据库初始化
    # ========================================================================

    def _init_db(self) -> None:
        """初始化 SQLite 事件数据库。

        创建 events 表，存储事件的完整字段。
        使用 check_same_thread=False 支持多线程写入。
        """
        self._db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                shelf_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sku_id TEXT NOT NULL,
                slot_id TEXT NOT NULL,
                track_id INTEGER,
                confidence REAL,
                snapshot_path TEXT
            )
        """)
        self._db_conn.commit()
        logger.info(f"事件数据库已就绪: {self.db_path}")

    # ========================================================================
    # 事件处理
    # ========================================================================

    def handle_event(self, event: ShelfEvent) -> None:
        """处理一个事件：截图保存 + 数据库写入 + WebSocket 广播。

        这是同步模式下的入口（管线循环在后台线程中运行时使用）。

        Args:
            event: 待处理的 ShelfEvent 实例
        """
        # ---- 1. 截图保存 ----
        snapshot_path = ""
        if self.save_snapshots and event.frame is not None:
            fname = f"{event.event_id}.jpg"
            fpath = self.snapshot_dir / fname
            cv2.imwrite(str(fpath), event.frame)
            snapshot_path = str(fpath)

        event.snapshot_path = snapshot_path

        # ---- 2. SQLite 持久化 ----
        if self._db_conn is not None:
            self._db_conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, event.timestamp, event.camera_id, event.shelf_id,
                 event.event_type, event.sku_id, event.slot_id, event.track_id,
                 event.confidence, snapshot_path),
            )
            self._db_conn.commit()

        # ---- 3. WebSocket 同步广播 ----
        self.broadcast(event)

    def broadcast(self, event: ShelfEvent) -> None:
        """向所有 WebSocket 客户端同步广播事件。

        线程安全：使用 _lock 保护客户端集合的读写。

        Args:
            event: 待广播的 ShelfEvent
        """
        data = json.dumps(self._event_to_dict(event), ensure_ascii=False)
        dead: set = set()  # 收集已断开的客户端

        with self._lock:
            for ws in self._ws_clients:
                try:
                    # WebSocket.send_text() 在同步模式下也能工作
                    ws.send_text(data)
                except Exception:
                    dead.add(ws)
            # 清理已断开的客户端
            self._ws_clients -= dead

    async def broadcast_async(self, event: ShelfEvent, websocket_clients: list) -> None:
        """异步向 WebSocket 客户端列表广播事件。

        用于 FastAPI 的 async 上下文（管线循环在 asyncio 中运行时使用）。

        Args:
            event: 待广播的 ShelfEvent
            websocket_clients: 当前连接的 WebSocket 客户端列表
        """
        from fastapi import WebSocket

        data = json.dumps(self._event_to_dict(event), ensure_ascii=False)
        dead = []
        for ws in websocket_clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        # 清理断开的客户端
        for ws in dead:
            websocket_clients.remove(ws)

    # ========================================================================
    # WebSocket 客户端管理
    # ========================================================================

    def register_ws(self, ws) -> None:
        """注册 WebSocket 客户端（同步模式）。"""
        with self._lock:
            self._ws_clients.add(ws)

    def unregister_ws(self, ws) -> None:
        """注销 WebSocket 客户端（同步模式）。"""
        with self._lock:
            self._ws_clients.discard(ws)

    @property
    def ws_client_count(self) -> int:
        """当前连接的 WebSocket 客户端数量。"""
        return len(self._ws_clients)

    # ========================================================================
    # 序列化
    # ========================================================================

    @staticmethod
    def _event_to_dict(event: ShelfEvent) -> dict[str, Any]:
        """将 ShelfEvent 序列化为 JSON 可用的字典。

        frame 字段（numpy 数组）不予序列化，只输出元数据和截图路径。
        """
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "camera_id": event.camera_id,
            "shelf_id": event.shelf_id,
            "event_type": event.event_type,
            "sku_id": event.sku_id,
            "slot_id": event.slot_id,
            "track_id": event.track_id,
            "confidence": event.confidence,
            "snapshot_path": event.snapshot_path,
        }

    # ========================================================================
    # 资源释放
    # ========================================================================

    def close(self) -> None:
        """关闭数据库连接，释放资源。"""
        if self._db_conn is not None:
            self._db_conn.close()
            logger.info(f"事件数据库已关闭")

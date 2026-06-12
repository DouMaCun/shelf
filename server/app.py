# -*- coding: utf-8 -*-
"""
Shelf Monitor 服务入口。

基于 FastAPI 的货架监控推理服务，提供：
- REST API：健康检查、货架状态查询、事件历史查询、事件截图获取
- WebSocket：实时事件推送
- 后台推理管线：视频输入 -> YOLO 检测 -> ByteTrack 跟踪 -> 货架状态 -> 事件判定 -> 输出

启动方式：
    python -m server.app -c config/default.yaml -s config/shelf_layout.yaml

启动选项：
    -c, --config       全局配置文件路径（默认 config/default.yaml）
    -s, --shelf        货架布局配置文件路径（默认 config/shelf_layout.yaml）
    --no-pipeline      仅启动 Web 服务，不启动推理管线（用于调试 API 接口）

架构说明：
    init_pipeline()  -> 创建所有管线组件（视频源、检测器、跟踪器、货架状态、事件引擎、输出）
    run_pipeline_loop() -> async 主循环，逐帧处理：读取 -> ROI裁剪 -> 检测 -> 跟踪 -> 状态更新 -> 事件判定 -> 推送
    lifespan() -> FastAPI 生命周期管理器，启动/关闭管线
"""

from __future__ import annotations

import argparse
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from loguru import logger

from pipeline.io.source import create_source, VideoSource
from pipeline.detector.yolo_detector import YOLODetector
from pipeline.detector.roi import apply_roi
from pipeline.tracker.byte_tracker import ByteTracker
from pipeline.shelf_state import ShelfState, ShelfConfig
from pipeline.event_engine import EventEngine, ShelfEvent
from pipeline.output import EventOutput


# ============================================================================
# 全局管线组件（模块级单例）
# ============================================================================

video_source: VideoSource | None = None
detector: YOLODetector | None = None
tracker: ByteTracker | None = None
shelf_state: ShelfState | None = None
event_engine: EventEngine | None = None
event_output: EventOutput | None = None

# WebSocket 客户端列表（在 asyncio 上下文中维护）
ws_clients: list[WebSocket] = []

# 管线运行控制
pipeline_task: asyncio.Task | None = None
pipeline_running = False

# 事件循环引用（用于跨线程调度）
loop: asyncio.AbstractEventLoop | None = None


# ============================================================================
# 配置加载与管线初始化
# ============================================================================

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。

    Args:
        config_path: YAML 文件路径

    Returns:
        dict: 解析后的配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_pipeline(config: dict, shelf_config: ShelfConfig) -> None:
    """初始化推理管线所有组件。

    按顺序创建以下组件：
    1. 视频输入源（VideoSource）
    2. YOLO 检测器（YOLODetector，同时加载 ONNX 模型）
    3. ByteTrack 跟踪器（ByteTracker）
    4. 货架状态管理器（ShelfState）
    5. 事件输出管理器（EventOutput）
    6. 事件判定引擎（EventEngine）

    Args:
        config: 全局配置字典
        shelf_config: 货架布局配置对象
    """
    global video_source, detector, tracker, shelf_state, event_engine, event_output

    pipeline_cfg = config['pipeline']
    output_cfg = config.get('output', {})

    # ---- 1. 视频输入源 ----
    video_source = create_source(pipeline_cfg['input'])
    logger.info("视频源已创建")

    # ---- 2. YOLO 检测器 ----
    detector = YOLODetector(
        model_path=pipeline_cfg['detector']['model_path'],
        conf_threshold=pipeline_cfg['detector'].get('conf_threshold', 0.5),
        iou_threshold=pipeline_cfg['detector'].get('iou_threshold', 0.45),
        input_size=tuple(pipeline_cfg['detector'].get('input_size', [640, 640])),
    )
    detector.load()  # 加载 ONNX 模型
    logger.info("检测器已加载")

    # ---- 3. ByteTrack 跟踪器 ----
    tracker_cfg = pipeline_cfg['tracker']
    tracker = ByteTracker(
        track_thresh=tracker_cfg.get('track_thresh', 0.5),
        match_thresh=tracker_cfg.get('match_thresh', 0.8),
        track_buffer=tracker_cfg.get('track_buffer', 30),
        frame_rate=tracker_cfg.get('frame_rate', 5),
    )
    logger.info("跟踪器已初始化")

    # ---- 4. 货架状态管理器 ----
    event_cfg = pipeline_cfg['event']
    shelf_state = ShelfState(
        config=shelf_config,
        min_iou_for_slot=event_cfg.get('min_iou_for_slot', 0.3),
        debounce_frames=event_cfg.get('debounce_frames', 5),
    )
    logger.info("货架状态已初始化")

    # ---- 5. 事件输出管理器 ----
    event_output = EventOutput(
        save_snapshots=output_cfg.get('save_snapshots', False),
        snapshot_dir=output_cfg.get('snapshot_dir', 'data/snapshots'),
        db_path=output_cfg.get('db_path', ''),
    )
    logger.info("事件输出已初始化")

    # ---- 6. 事件判定引擎 ----
    # 不设置同步回调（on_event），因为在 async 管线循环中异步广播
    event_engine = EventEngine(
        camera_id=shelf_config.camera_id,
        shelf_id=shelf_config.shelf_id,
        on_event=None,
    )
    logger.info("事件引擎已初始化")


# ============================================================================
# 推理管线主循环
# ============================================================================

async def run_pipeline_loop(config: dict) -> None:
    """推理管线主循环（异步）。

    每帧的处理流程：
    1. 从视频源缓冲区取一帧
    2. 可选：ROI 裁剪，只保留货架区域
    3. YOLO 目标检测
    4. ByteTrack 多目标跟踪
    5. 更新货架槽位状态（含去抖）
    6. 事件判定（比较前后帧状态差异）
    7. WebSocket 广播新事件

    每 100 帧输出一次性能统计（帧号、跟踪数、推理耗时、实际帧率）。

    Args:
        config: 全局配置字典
    """
    global video_source, detector, tracker, shelf_state, event_engine, event_output
    global pipeline_running

    pipeline_cfg = config['pipeline']
    roi_cfg = pipeline_cfg['detector'].get('roi', [0, 0, 1, 1])  # 默认全图

    # 启动视频源（后台线程开始读取帧）
    video_source.start()
    pipeline_running = True
    frame_count = 0

    try:
        while pipeline_running and video_source._running:
            # ---- Step 1: 取帧 ----
            frame = video_source.read()
            if frame is None:
                # 缓冲区空，等待后台线程填充
                await asyncio.sleep(0.01)
                continue

            frame_count += 1

            # ---- Step 2: ROI 裁剪 ----
            if roi_cfg != [0, 0, 1, 1]:
                roi_frame = apply_roi(frame, roi_cfg)
            else:
                roi_frame = frame

            # ---- Step 3: 目标检测 ----
            detections = detector.detect(roi_frame)

            # ---- Step 4: 多目标跟踪 ----
            tracks = tracker.update(detections)

            # ---- Step 5: 货架状态更新 ----
            now = time.time()
            slot_occupancy = shelf_state.update(tracks, now)

            # ---- Step 6: 事件判定 ----
            new_events = event_engine.update(slot_occupancy, shelf_state, frame)

            # ---- Step 7: WebSocket 广播新事件 ----
            if event_output and new_events:
                for evt in new_events:
                    await event_output.broadcast_async(evt, ws_clients)

            # 每 100 帧输出一次性能统计
            if frame_count % 100 == 0:
                logger.debug(
                    f"Frame={frame_count}, "
                    f"tracks={len(tracks)}, "
                    f"inference={detector.avg_inference_time * 1000:.1f}ms, "
                    f"fps={video_source.actual_fps:.1f}"
                )

            # 短暂让出控制权，避免阻塞 asyncio 事件循环
            await asyncio.sleep(0)

    except Exception as e:
        logger.exception(f"管线异常: {e}")
    finally:
        video_source.stop()
        pipeline_running = False
        logger.info("管线主循环已退出")


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理器。

    应用启动时：保存事件循环引用
    应用关闭时：停止管线、释放资源
    """
    global loop
    loop = asyncio.get_event_loop()
    logger.info("服务启动中...")
    yield
    # ---- 关闭操作 ----
    global pipeline_running
    pipeline_running = False
    if video_source:
        video_source.stop()
    if event_output:
        event_output.close()
    logger.info("服务已停止")


# 创建 FastAPI 应用实例
app = FastAPI(
    title="Shelf Monitor - 无人货架监控",
    version="0.1.0",
    description="基于 YOLO + ByteTrack 的视觉货架商品取放识别系统",
    lifespan=lifespan,
)


# ============================================================================
# REST API 路由
# ============================================================================

@app.get("/health")
async def health():
    """健康检查接口。

    返回当前服务的运行状态和性能指标：
    - pipeline_running: 推理管线是否在运行
    - ws_clients: 当前 WebSocket 连接数
    - fps: 实际视频源帧率
    - inference_ms: 检测器平均推理耗时（毫秒）
    """
    return {
        "status": "ok",
        "pipeline_running": pipeline_running,
        "ws_clients": len(ws_clients),
        "fps": video_source.actual_fps if video_source else 0,
        "inference_ms": detector.avg_inference_time * 1000 if detector else 0,
    }


@app.get("/api/shelf/state")
async def get_shelf_state():
    """获取当前货架所有槽位的状态快照。

    返回每个槽位的：
    - slot_id, sku_id
    - occupied: 是否被商品占据
    - confidence: 检测置信度
    - last_seen: 最后检测时间戳

    用于业务后端或前端监控面板展示货架实时状态。
    """
    if shelf_state is None:
        return JSONResponse({"error": "管线未初始化"}, status_code=503)
    return shelf_state.get_snapshot()


@app.get("/api/events")
async def get_events(since: float | None = None):
    """查询历史事件。

    Args:
        since: Unix 时间戳，只返回此时间之后的事件。不传返回全部。

    Returns:
        事件列表，每项包含：event_id, timestamp, event_type, sku_id, slot_id, confidence
    """
    if event_engine is None:
        return JSONResponse({"error": "管线未初始化"}, status_code=503)
    events = event_engine.get_events(since)
    return [
        {
            "event_id": e.event_id,
            "timestamp": e.timestamp,
            "event_type": e.event_type,
            "sku_id": e.sku_id,
            "slot_id": e.slot_id,
            "confidence": e.confidence,
        }
        for e in events
    ]


@app.get("/api/snapshot/{event_id}")
async def get_snapshot(event_id: str):
    """获取事件截图。

    返回事件发生时的帧图像 JPEG 文件。
    仅当 output 配置中 save_snapshots: true 时可用。

    Args:
        event_id: 事件 ID
    """
    snapshot_dir = Path("data/snapshots")
    fpath = snapshot_dir / f"{event_id}.jpg"
    if not fpath.exists():
        return JSONResponse({"error": "截图不存在"}, status_code=404)
    return FileResponse(str(fpath))


# ============================================================================
# WebSocket 端点
# ============================================================================

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket 实时事件推送端点。

    客户端连接到此端点后，将实时接收 pick/place 事件的 JSON 推送。
    事件格式与 /api/events 返回的单项结构一致。

    连接生命周期：
    - 客户端连接 -> accept() -> 加入 ws_clients 列表
    - 保持连接 -> receive_text() 阻塞等待（检测断开）
    - 客户端断开 -> WebSocketDisconnect 异常 -> 从列表移除
    """
    await websocket.accept()
    ws_clients.append(websocket)
    logger.info(f"WebSocket 客户端已连接, 当前共 {len(ws_clients)} 个连接")
    try:
        # 保持连接直到客户端断开
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.remove(websocket)
        logger.info(f"WebSocket 客户端已断开, 剩余 {len(ws_clients)} 个连接")


# ============================================================================
# 程序入口
# ============================================================================

def main():
    """程序主入口。

    流程：
    1. 解析命令行参数
    2. 配置 loguru 日志（控制台 + 文件轮转）
    3. 加载全局配置和货架布局配置
    4. 初始化推理管线
    5. 启动推理管线 async task（可选 --no-pipeline 跳过）
    6. 启动 uvicorn Web 服务
    """
    parser = argparse.ArgumentParser(description="Shelf Monitor - 无人货架监控服务")
    parser.add_argument("-c", "--config", default="config/default.yaml",
                        help="全局配置文件路径（默认 config/default.yaml）")
    parser.add_argument("-s", "--shelf", default="config/shelf_layout.yaml",
                        help="货架布局配置文件路径（默认 config/shelf_layout.yaml）")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="仅启动 Web 服务，不启动推理管线（用于 API 调试）")
    args = parser.parse_args()

    # 配置日志：控制台 + 文件轮转
    logger.add(
        "logs/pipeline_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )

    # 加载配置
    config = load_config(args.config)
    shelf_config = ShelfConfig.from_yaml(args.shelf)

    # 初始化推理管线
    init_pipeline(config, shelf_config)

    # 启动推理管线（在 asyncio 事件循环中作为 task 运行）
    if not args.no_pipeline:
        global pipeline_task
        loop = asyncio.get_event_loop()
        pipeline_task = loop.create_task(run_pipeline_loop(config))
        logger.info("推理管线 task 已创建")

    # 启动 Web 服务
    server_cfg = config['server']
    uvicorn.run(
        app,
        host=server_cfg.get('host', '0.0.0.0'),
        port=server_cfg.get('port', 8000),
        log_level=config.get('logging', {}).get('level', 'info').lower(),
    )


if __name__ == "__main__":
    main()

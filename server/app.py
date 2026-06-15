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
from pipeline.detector.rknn_detector import RKNNDetector
from pipeline.interaction_monitor import InteractionMonitor, InteractionState
from pipeline.identifier.item_classifier import ItemClassifier
from pipeline.event_engine import EventEngine, ShelfEvent
from pipeline.output import EventOutput


# ============================================================================
# 全局管线组件（模块级单例）
# ============================================================================

video_source: VideoSource | None = None
detector = None          # 手臂/人体检测器（COCO 预训练，YOLODetector 或 RKNNDetector）
item_detector = None     # 商品分类检测器（自定义训练）
interaction_monitor: InteractionMonitor | None = None
item_classifier: ItemClassifier | None = None
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


def _make_detector(det_cfg: dict):
    """根据配置创建检测器（ONNX 或 RKNN）。"""
    backend = det_cfg.get('backend', 'onnx')
    kwargs = dict(
        model_path=det_cfg['model_path'],
        conf_threshold=det_cfg.get('conf_threshold', 0.5),
        iou_threshold=det_cfg.get('iou_threshold', 0.45),
        input_size=tuple(det_cfg.get('input_size', [640, 640])),
    )
    if backend == 'rknn':
        d = RKNNDetector(**kwargs)
        logger.info(f"检测器后端: RKNN NPU ({det_cfg['model_path']})")
    else:
        d = YOLODetector(**kwargs)
        logger.info(f"检测器后端: ONNX Runtime ({det_cfg['model_path']})")
    d.load()
    return d


def init_pipeline(config: dict) -> None:
    """初始化出口区识别管线所有组件。

    组件创建顺序：
    1. 视频输入源（VideoSource）
    2. 手臂检测器（COCO 预训练，用于检测手臂触发交互）
    3. 商品检测器（自定义训练，用于 YOLO 视觉分类）
    4. 交互状态机（InteractionMonitor）
    5. 商品分类器（ItemClassifier，封装商品检测器）
    6. 事件输出管理器（EventOutput）
    7. 事件引擎（EventEngine）

    Args:
        config: 全局配置字典（来自 default.yaml）
    """
    global video_source, detector, item_detector
    global interaction_monitor, item_classifier
    global event_engine, event_output

    pipeline_cfg = config['pipeline']
    interaction_cfg = config.get('interaction', {})
    output_cfg = config.get('output', {})

    # ---- 1. 视频输入源 ----
    video_source = create_source(pipeline_cfg['input'])
    logger.info("视频源已创建")

    # ---- 2. 手臂检测器（COCO 预训练，检测 person 类触发交互）----
    detector = _make_detector(pipeline_cfg['detector'])

    # ---- 3. 商品检测器（自定义训练，用于 YOLO 视觉分类回退）----
    item_det_cfg = config.get('item_classifier', {})
    if item_det_cfg.get('model_path'):
        item_detector = _make_detector(item_det_cfg)
    else:
        # 未配置独立商品模型时，复用手臂检测器（开发调试用）
        item_detector = detector
        logger.warning("item_classifier.model_path 未配置，视觉分类将复用手臂检测器")

    # ---- 4. 交互状态机 ----
    interaction_monitor = InteractionMonitor(
        cooldown_seconds=interaction_cfg.get('cooldown_seconds', 3.0),
        frame_buffer_size=interaction_cfg.get('frame_buffer_size', 30),
    )
    logger.info("交互状态机已初始化")

    # ---- 5. 商品分类器 ----
    mapping_path = item_det_cfg.get('model_mapping', 'config/model_mapping.yaml')
    item_classifier = ItemClassifier(
        detector=item_detector,
        mapping_path=mapping_path,
    )
    logger.info("商品分类器已初始化")

    # ---- 6. 事件输出管理器 ----
    event_output = EventOutput(
        save_snapshots=output_cfg.get('save_snapshots', False),
        snapshot_dir=output_cfg.get('snapshot_dir', 'data/snapshots'),
        db_path=output_cfg.get('db_path', ''),
    )
    logger.info("事件输出已初始化")

    # ---- 7. 事件引擎 ----
    camera_id = config.get('camera_id', 'cam_01')
    shelf_id = config.get('shelf_id', 'shelf_01')
    event_engine = EventEngine(camera_id=camera_id, shelf_id=shelf_id)
    logger.info("事件引擎已初始化")


# ============================================================================
# 推理管线主循环
# ============================================================================

async def run_pipeline_loop(config: dict) -> None:
    """推理管线主循环——出口区识别模式（异步）。

    每帧的处理流程：
    1. 根据当前状态决定采样间隔（IDLE 低频 / ACTIVE 高频）
    2. 从视频源取帧
    3. 手臂检测（YOLO，判断是否有人伸手进柜）
    4. 推进交互状态机
    5. 状态变为 CLASSIFYING 时：取最清晰帧 → YOLO 视觉分类 → 发出事件

    Args:
        config: 全局配置字典
    """
    global video_source, detector, interaction_monitor
    global item_classifier, event_engine, event_output
    global pipeline_running

    interaction_cfg = config.get('interaction', {})
    person_class_id = interaction_cfg.get('person_class_id', 0)
    idle_fps = config['pipeline']['input'].get('fps', 5)
    active_fps = interaction_cfg.get('active_fps', 10)
    idle_interval = 1.0 / max(idle_fps, 1)
    active_interval = 1.0 / max(active_fps, 1)

    video_source.start()
    pipeline_running = True
    frame_count = 0
    prev_state = InteractionState.IDLE

    try:
        while pipeline_running and video_source._running:
            # 根据状态选采样间隔
            current_state = interaction_monitor.state
            interval = active_interval if current_state == InteractionState.ACTIVE else idle_interval

            # ---- Step 1: 取帧 ----
            frame = video_source.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1

            # ---- Step 2: 手臂检测 ----
            detections = detector.detect(frame)
            has_person = any(
                int(d[5]) == person_class_id for d in detections
            ) if len(detections) > 0 else False

            # ---- Step 3: 推进状态机 ----
            new_state = interaction_monitor.update(frame, has_person)

            # ---- Step 4: 刚进入 CLASSIFYING → YOLO 识别并发事件 ----
            if new_state == InteractionState.CLASSIFYING and prev_state != InteractionState.CLASSIFYING:
                best_frame = interaction_monitor.get_best_frame()
                sku_id, confidence = _identify_item(best_frame)
                event = event_engine.emit_pick(sku_id, confidence, best_frame)
                if event_output:
                    await event_output.broadcast_async(event, ws_clients)
                interaction_monitor.complete_classification()

            prev_state = new_state

            # 每 200 帧输出一次性能统计
            if frame_count % 200 == 0:
                logger.debug(
                    f"Frame={frame_count}, state={current_state.value}, "
                    f"inference={detector.avg_inference_time * 1000:.1f}ms, "
                    f"fps={video_source.actual_fps:.1f}"
                )

            await asyncio.sleep(interval)

    except Exception as e:
        logger.exception(f"管线异常: {e}")
    finally:
        video_source.stop()
        pipeline_running = False
        logger.info("管线主循环已退出")


def _identify_item(frame) -> tuple[str | None, float]:
    """YOLO 视觉分类识别被取走商品。

    Args:
        frame: 最佳帧（BGR ndarray），可能为 None

    Returns:
        (sku_id, confidence)：识别结果
    """
    if frame is None:
        logger.warning("无有效帧可用于识别")
        return None, 0.0
    return item_classifier.classify(frame)


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
    for det in (detector, item_detector):
        if det and hasattr(det, 'release'):
            det.release()
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
    """获取当前交互状态和最近事件摘要。

    返回：
    - interaction_state: 当前状态机状态（idle / active / classifying / cooldown）
    - buffer_frames: ACTIVE 阶段已缓冲帧数
    - recent_events: 最近 10 条 pick 事件
    """
    if interaction_monitor is None:
        return JSONResponse({"error": "管线未初始化"}, status_code=503)
    recent = event_engine.get_events() if event_engine else []
    return {
        "interaction_state": interaction_monitor.state.value,
        "buffer_frames": interaction_monitor.buffer_size,
        "recent_events": [
            {"event_id": e.event_id, "timestamp": e.timestamp,
             "sku_id": e.sku_id, "confidence": e.confidence}
            for e in recent[-10:]
        ],
    }


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

    # 初始化推理管线
    init_pipeline(config)

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

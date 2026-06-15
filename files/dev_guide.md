# Shelf Monitor 二次开发指引

本文档面向需要在此项目基础上进行二次开发的工程师，涵盖模块 API、扩展点、配置说明和对接接口。

---

## 目录

1. [模块地图](#一模块地图)
2. [核心模块 API](#二核心模块-api)
3. [常见扩展场景](#三常见扩展场景)
4. [完整配置参考](#四完整配置参考)
5. [REST API / WebSocket 接口](#五rest-api--websocket-接口)
6. [数据库 Schema](#六数据库-schema)

---

## 一、模块地图

```
pipeline/
  io/
    source.py          VideoSource（ABC）+ FileSource / USBSource / RTSPSource + create_source()
    preprocess.py      preprocess_for_yolo()  帧预处理工具函数

  detector/
    yolo_detector.py   YOLODetector    ONNX Runtime 推理
    rknn_detector.py   RKNNDetector    RK3588 NPU 推理（接口与 YOLODetector 相同）
    roi.py             apply_roi()     ROI 裁剪工具

  interaction_monitor.py  InteractionMonitor  4 状态机 + 帧缓冲 + 最清晰帧选择

  identifier/
    item_classifier.py ItemClassifier  YOLO 视觉分类 → (sku_id, confidence)

  event_engine.py      ShelfEvent（dataclass）+ EventEngine
  output.py            EventOutput  截图 / SQLite / WebSocket 广播
  shelf_state.py       V1 槽位状态（保留，V2 管线不使用）

server/
  app.py               FastAPI 应用 + 管线入口

config/
  default.yaml         全局配置（参见第四节）
  model_mapping.yaml   class_id → sku_id 映射
```

---

## 二、核心模块 API

### 2.1 VideoSource（`pipeline/io/source.py`）

```python
src = create_source(config_dict)   # 工厂函数，根据 type 字段创建实例

src.start()                        # 启动后台读帧线程
frame = src.read()                 # 取一帧（非阻塞，无帧时返回 None）
src.stop()                         # 停止线程，释放资源

src.actual_fps   # float  实际帧率（从启动到现在的均值）
src._running     # bool   线程是否在运行
```

`config_dict` 必填字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | str | `"file"` / `"usb"` / `"rtsp"` |
| `source` | str | 文件路径、摄像头索引（"0"）或 RTSP URL |
| `fps` | int | 目标帧率，默认 5 |
| `width` / `height` | int | 输出帧尺寸 |
| `buffer_size` | int | 缓冲队列大小，默认 30 |

---

### 2.2 YOLODetector / RKNNDetector（`pipeline/detector/`）

两个检测器接口完全相同，可以互换。

```python
det = YOLODetector(
    model_path="models/yolo11n.onnx",
    conf_threshold=0.5,
    iou_threshold=0.45,
    input_size=(640, 640),
)
det.load()                       # 加载模型（必须在 detect() 前调用）
detections = det.detect(frame)   # 返回 ndarray (N, 6)
                                 # 每行: [x1, y1, x2, y2, confidence, class_id]
                                 # 无检测时返回 shape (0, 6) 的空数组

det.avg_inference_time           # float  最近 50 次推理的平均耗时（秒）
```

---

### 2.3 InteractionMonitor（`pipeline/interaction_monitor.py`）

```python
monitor = InteractionMonitor(
    cooldown_seconds=3.0,   # 识别完成后静默时长
    frame_buffer_size=30,   # ACTIVE 阶段最多缓冲帧数
)

# 每帧调用
state = monitor.update(frame, has_person)   # → InteractionState 枚举

# 状态 == CLASSIFYING 时
best_frame = monitor.get_best_frame()       # → ndarray | None（Laplacian 方差最大帧）

# 识别完成后
monitor.complete_classification()           # 进入 COOLDOWN，N 秒后自动回 IDLE

# 属性
monitor.state          # InteractionState  当前状态
monitor.buffer_size    # int  当前缓冲帧数量
monitor.reset()        # 强制重置到 IDLE（异常恢复用）
```

**状态枚举**：`InteractionState.IDLE / ACTIVE / CLASSIFYING / COOLDOWN`

---

### 2.4 ItemClassifier（`pipeline/identifier/item_classifier.py`）

```python
classifier = ItemClassifier(
    detector=det,                             # 已 load() 的 YOLODetector 或 RKNNDetector
    mapping_path="config/model_mapping.yaml",
)

sku_id, confidence = classifier.classify(frame)
# sku_id: str | None  识别结果，无法识别时为 None
# confidence: float   检测置信度 [0, 1]
```

**model_mapping.yaml 格式**：

```yaml
mapping:
  0: {sku_id: "resistor_100ohm", name: "100Ω 电阻"}
  1: {sku_id: "capacitor_10uf",  name: "10μF 电容"}
  2: {sku_id: "led_red_5mm",     name: "红色 LED 5mm"}
```

---

### 2.5 EventEngine（`pipeline/event_engine.py`）

```python
engine = EventEngine(
    camera_id="cam_01",
    shelf_id="shelf_01",
    on_event=my_callback,      # 可选，每次产生事件时调用 fn(event: ShelfEvent)
)

# V2 出口区识别模式（主要用法）
event = engine.emit_pick(sku_id, confidence, frame)   # → ShelfEvent

# 查询历史
events = engine.get_events()                  # 全部事件
events = engine.get_events(since=timestamp)   # Unix 时间戳之后的事件

engine.reset()   # 清空事件历史（数据源切换时调用）
```

**ShelfEvent 字段**：

```python
@dataclass
class ShelfEvent:
    event_id: str          # "evt_" + 12 位随机 hex
    timestamp: str         # ISO 8601 格式，如 "2026-06-15T10:23:45"
    camera_id: str
    shelf_id: str
    event_type: str        # "pick" | "place"
    sku_id: str            # 识别到的 SKU，无法识别时为 "unknown"
    slot_id: str           # V2 出口区模式固定为 ""
    track_id: int          # V2 出口区模式固定为 -1
    confidence: float      # [0, 1]
    snapshot_path: str     # 截图路径，EventOutput 处理后填充
    frame: ndarray | None  # 原始帧图像，不序列化
```

---

### 2.6 EventOutput（`pipeline/output.py`）

```python
output = EventOutput(
    save_snapshots=True,
    snapshot_dir="data/snapshots",
    db_path="data/events.db",   # 空字符串表示不启用 SQLite
)

# 异步广播（FastAPI asyncio 上下文）
await output.broadcast_async(event, ws_clients_list)

# 同步处理（截图 + SQLite + 广播）
output.handle_event(event)

output.close()   # 关闭数据库连接（服务关闭时调用）
```

---

## 三、常见扩展场景

### 3.1 接入新摄像头类型

继承 `VideoSource`，实现两个抽象方法：

```python
# pipeline/io/source.py 内或新文件中
class HTTPStreamSource(VideoSource):
    def _open(self) -> cv2.VideoCapture:
        # 用 ffmpeg 或 requests 打开 HTTP 视频流
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

    def _read_frame(self) -> np.ndarray | None:
        ret, frame = self._capture.read()
        return frame if ret else None
```

然后在 `create_source()` 的 `if/elif` 中注册：

```python
elif source_type == "http":
    return HTTPStreamSource(source_addr, **kwargs)
```

配置：

```yaml
pipeline:
  input:
    type: http
    source: "http://192.168.1.100/video"
```

---

### 3.2 接入新推理后端（如 OpenVINO、TensorRT）

实现与 `YOLODetector` 相同的接口：

```python
class OpenVINODetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45, input_size=(640, 640)):
        ...

    def load(self) -> None:
        # 加载 OpenVINO IR 模型
        ...

    def detect(self, frame: np.ndarray) -> np.ndarray:
        # 推理，返回 ndarray (N, 6): [x1, y1, x2, y2, conf, class_id]
        ...

    @property
    def avg_inference_time(self) -> float:
        ...
```

在 `server/app.py` 的 `_make_detector()` 中注册：

```python
def _make_detector(det_cfg: dict):
    backend = det_cfg.get('backend', 'onnx')
    ...
    elif backend == 'openvino':
        d = OpenVINODetector(**kwargs)
    ...
```

配置切换：

```yaml
detector:
  backend: openvino
  model_path: "models/yolo11n.xml"
```

---

### 3.3 替换商品识别逻辑

`_identify_item()` 是识别的入口点（`server/app.py`），替换它即可接入任意识别方案：

```python
# 示例：调用外部 HTTP 识别服务
def _identify_item(frame) -> tuple[str | None, float]:
    if frame is None:
        return None, 0.0

    # 编码为 JPEG，发送给外部服务
    _, buf = cv2.imencode(".jpg", frame)
    resp = requests.post(
        "http://ai-service/classify",
        files={"image": buf.tobytes()},
        timeout=2.0,
    )
    if resp.ok:
        data = resp.json()
        return data["sku_id"], data["confidence"]
    return None, 0.0
```

或替换为自定义 Python 分类逻辑（规则引擎、颜色直方图等）：

```python
def _identify_item(frame) -> tuple[str | None, float]:
    if frame is None:
        return None, 0.0
    # 你的识别逻辑
    return sku_id, confidence
```

---

### 3.4 新增事件输出通道

在 `server/app.py` 的管线循环中，`event_engine.emit_pick()` 返回 `ShelfEvent` 对象，可以在广播之后立即接入任意处理逻辑：

```python
event = event_engine.emit_pick(sku_id, confidence, best_frame)

# 原有推送
if event_output:
    await event_output.broadcast_async(event, ws_clients)

# 新增：发送到 MQTT
mqtt_client.publish("shelf/events", json.dumps({
    "sku_id": event.sku_id,
    "confidence": event.confidence,
    "timestamp": event.timestamp,
}))

# 新增：调用业务后端 HTTP 接口
async with aiohttp.ClientSession() as session:
    await session.post("http://erp/api/inventory/pick", json={
        "camera_id": event.camera_id,
        "sku_id": event.sku_id,
        "timestamp": event.timestamp,
    })
```

也可以利用 `EventEngine` 的回调机制，在 `init_pipeline()` 中注入：

```python
def my_event_handler(event: ShelfEvent):
    # 同步回调，在 asyncio 线程中调用，注意线程安全
    logger.info(f"业务处理: {event.sku_id}")

event_engine = EventEngine(
    camera_id=camera_id,
    shelf_id=shelf_id,
    on_event=my_event_handler,
)
```

---

### 3.5 新增 REST API 端点

直接在 `server/app.py` 中添加 FastAPI 路由：

```python
@app.get("/api/stats/today")
async def get_today_stats():
    """今日取货统计。"""
    events = event_engine.get_events() if event_engine else []
    today = time.strftime("%Y-%m-%d")
    today_events = [e for e in events if e.timestamp.startswith(today)]
    sku_counts = {}
    for e in today_events:
        sku_counts[e.sku_id] = sku_counts.get(e.sku_id, 0) + 1
    return {"date": today, "total": len(today_events), "by_sku": sku_counts}
```

---

### 3.6 多路摄像头支持

当前管线是单路架构（全局单例）。接入多路摄像头有两种方案：

**方案 A（推荐）：多进程，每路启动独立服务实例，端口不同**

```bash
python -m server.app -c config/cam_01.yaml  # 端口 8001
python -m server.app -c config/cam_02.yaml  # 端口 8002
```

每个 YAML 配置不同的 `camera_id`、`source`、`port`，上层聚合服务负责汇总事件。

**方案 B：单进程多任务，在 `main()` 中启动多个 `run_pipeline_loop()` asyncio task**，共享同一 FastAPI 实例。每路需要独立的组件集合（`video_source`、`detector`、`interaction_monitor` 等），可封装为 `Pipeline` 数据类避免全局变量冲突。

---

### 3.7 调整状态机行为

`InteractionMonitor` 的状态转换逻辑在 `update()` 方法中，可直接修改：

```python
# 示例：ACTIVE 超时保护（手臂一直在视野内也强制进入 CLASSIFYING）
MAX_ACTIVE_SECONDS = 10.0

elif self._state == InteractionState.ACTIVE:
    self._frame_buffer.append(frame.copy())
    if not has_person or (time.time() - self._active_since) > MAX_ACTIVE_SECONDS:
        self._state = InteractionState.CLASSIFYING
```

---

## 四、完整配置参考

`config/default.yaml` 所有字段说明：

```yaml
pipeline:
  input:
    type: file          # rtsp | usb | file
    source: ""          # 视频源地址（文件路径 / 摄像头索引 / RTSP URL）
    fps: 5              # 推理帧率（IDLE 状态下的采样频率）
    width: 1280         # 输出帧宽度（像素）
    height: 720         # 输出帧高度（像素）
    buffer_size: 30     # 帧缓冲队列大小

  detector:             # 手臂/人体检测器（COCO 预训练）
    backend: onnx       # onnx（x86 开发机）| rknn（RK3588 设备）
    model_path: ""      # ONNX 或 RKNN 模型路径
    conf_threshold: 0.5 # 置信度阈值（低于此值的检测结果被过滤）
    iou_threshold: 0.45 # NMS IoU 阈值
    roi: [0.0, 0.0, 1.0, 1.0]   # 感兴趣区域（归一化坐标 [x1,y1,x2,y2]）
    input_size: [640, 640]        # 模型输入尺寸

item_classifier:        # 商品识别器（自定义俯拍训练模型）
  backend: onnx
  model_path: ""        # 留空则复用 detector 模型（适合调试）
  conf_threshold: 0.5
  iou_threshold: 0.45
  input_size: [640, 640]
  model_mapping: "config/model_mapping.yaml"

interaction:
  person_class_id: 0       # COCO person 类 ID（0 为标准值，勿改）
  active_fps: 10           # ACTIVE 状态下的高频采样帧率
  frame_buffer_size: 30    # ACTIVE 阶段最多缓冲帧数（影响内存和最清晰帧选择范围）
  cooldown_seconds: 3      # 每次识别后的静默时长（秒），防止重复触发

camera_id: "cam_01"    # 写入事件日志的摄像头标识
shelf_id: "shelf_01"   # 写入事件日志的货架标识

server:
  host: "0.0.0.0"
  port: 8000
  ws_path: "/ws/events"
  cors_origins: ["*"]   # 允许跨域的来源列表

logging:
  level: "INFO"         # DEBUG | INFO | WARNING | ERROR
  rotation: "10 MB"     # 日志文件轮转大小
  retention: "7 days"   # 日志保留时长

output:
  save_snapshots: true             # 是否保存事件截图到磁盘
  snapshot_dir: "data/snapshots"   # 截图保存目录
  db_path: "data/events.db"        # SQLite 路径（空字符串 = 不启用）
```

---

## 五、REST API / WebSocket 接口

### REST API

| Method | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，返回管线状态和性能指标 |
| GET | `/api/shelf/state` | 当前交互状态和最近 10 条事件摘要 |
| GET | `/api/events?since=<unix_ts>` | 查询历史事件，`since` 不传则返回全部 |
| GET | `/api/snapshot/{event_id}` | 获取事件截图（JPEG），需开启 `save_snapshots` |

**`/health` 响应示例**：

```json
{
  "status": "ok",
  "pipeline_running": true,
  "ws_clients": 2,
  "fps": 4.9,
  "inference_ms": 18.3
}
```

**`/api/events` 单条事件格式**：

```json
{
  "event_id": "evt_a3f2b1c0d4e5",
  "timestamp": "2026-06-15T10:23:45",
  "event_type": "pick",
  "sku_id": "resistor_100ohm",
  "slot_id": "",
  "confidence": 0.87
}
```

### WebSocket

连接地址：`ws://<host>:<port>/ws/events`

连接建立后，每次发生 pick 事件时服务端推送 JSON：

```json
{
  "event_id": "evt_a3f2b1c0d4e5",
  "timestamp": "2026-06-15T10:23:45",
  "camera_id": "cam_01",
  "shelf_id": "shelf_01",
  "event_type": "pick",
  "sku_id": "resistor_100ohm",
  "slot_id": "",
  "track_id": -1,
  "confidence": 0.87,
  "snapshot_path": "data/snapshots/evt_a3f2b1c0d4e5.jpg"
}
```

**客户端连接示例（Python）**：

```python
import asyncio, websockets, json

async def listen():
    async with websockets.connect("ws://localhost:8000/ws/events") as ws:
        async for msg in ws:
            event = json.loads(msg)
            print(f"[{event['timestamp']}] {event['sku_id']} 被取走"
                  f"（置信度 {event['confidence']:.0%}）")

asyncio.run(listen())
```

---

## 六、数据库 Schema

启用 `output.db_path` 后，事件写入 SQLite 的 `events` 表：

```sql
CREATE TABLE events (
    event_id     TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    camera_id    TEXT NOT NULL,
    shelf_id     TEXT NOT NULL,
    event_type   TEXT NOT NULL,    -- "pick" | "place"
    sku_id       TEXT NOT NULL,
    slot_id      TEXT NOT NULL,    -- V2 出口区模式固定为空字符串
    track_id     INTEGER,          -- V2 出口区模式固定为 -1
    confidence   REAL,
    snapshot_path TEXT
);
```

**常用查询**：

```sql
-- 今日取货统计
SELECT sku_id, COUNT(*) AS cnt
FROM events
WHERE timestamp LIKE '2026-06-15%' AND event_type = 'pick'
GROUP BY sku_id
ORDER BY cnt DESC;

-- 查询某 SKU 的最近 10 条记录
SELECT event_id, timestamp, confidence, snapshot_path
FROM events
WHERE sku_id = 'resistor_100ohm'
ORDER BY timestamp DESC
LIMIT 10;

-- 按小时统计取货量（流量分析）
SELECT strftime('%H', timestamp) AS hour, COUNT(*) AS cnt
FROM events
WHERE event_type = 'pick'
GROUP BY hour
ORDER BY hour;
```

---

> 阅读完本文档后，建议从 `server/app.py` 的 `run_pipeline_loop()` 入手，结合 `config/default.yaml` 启动一个最小化实例，再逐步替换需要定制的模块。

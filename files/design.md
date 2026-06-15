## ⚠️ V2 架构更新说明

> 本文档（第一至十二节）描述的是**初始槽位方案（V1）**，已被 V2 架构取代。
> V1 假设摄像头正对货架正面、商品固定摆放，不适用于俯拍 + 随机摆放场景。
> 保留原文作为设计背景和历史参考。

### V2 核心变化

**场景**：摄像头装在柜门顶部向下俯拍，商品（大件电子器件）随机摆放，无门磁传感器。

**关键洞察**：任何被取走的商品都必须经过柜门开口（摄像头正下方），取出时商品会举起穿过视野，这是最佳识别窗口。

**V2 管线**（替代 V1 的槽位状态机）：

```
摄像头帧
  ↓
手臂检测（YOLO COCO person 类）
  ↓
交互状态机（InteractionMonitor）
  IDLE → ACTIVE（缓冲帧）→ CLASSIFYING → COOLDOWN → IDLE
  ↓（进入 CLASSIFYING 时）
选最清晰帧（Laplacian 方差）
  ↓
YOLO 视觉分类（需俯拍角度训练数据）
  ↓
pick 事件 → WebSocket 推送 / 截图 / SQLite
```

**V2 新增/改变的文件**：

| 文件 | 变化 |
|---|---|
| `pipeline/interaction_monitor.py` | 新增，替代 `shelf_state.py` |
| `pipeline/identifier/item_classifier.py` | 新增，YOLO 视觉分类 |
| `pipeline/event_engine.py` | 新增 `emit_pick()` 方法 |
| `server/app.py` | 重写 `init_pipeline()` 和 `run_pipeline_loop()` |
| `config/default.yaml` | 新增 `interaction` 和 `item_classifier` 节 |
| `config/shelf_layout.yaml` | V2 不再使用（商品随机摆放） |

**新增依赖**：无（复用现有依赖栈）

---

## 一、项目概述

### 1.1 业务场景

无人货架 / 视觉货架场景下，通过摄像头实时监控货架区域，识别顾客对商品的**取走（pick）** 和**放回（place）** 行为，支撑自动结算、库存管理、异常检测等下游业务。

### 1.2 核心目标

- 实时检测货架上每个槽位的占用状态变化（数量级计数为后续版本目标，见 6.6）。
- 区分 pick（取走）、place（放回）、misplace（错位放回）三种事件。
- 输出结构化事件流：`{timestamp, event_type, sku_id, slot_id, confidence}`。

### 1.3 技术选型

| 层级 | 技术 | 理由 |
|------|------|------|
| 目标检测 | YOLOv8 / YOLOv11 | 成熟、推理快、社区活跃 |
| 目标跟踪（可选） | ByteTrack | MVP 阶段不启用，后续用于 pick 二次确认，见 6.4 |
| 货架状态建模 | RoI 分区 + 时空关联 | 将检测框和货架槽位绑定 |
| 事件判定 | 状态机（规则引擎） | 基于槽位占用状态转移判定 pick/place |
| 推理框架 | ONNX Runtime / OpenVINO | 跨平台部署，CPU/iGPU 友好 |
| 应用框架 | FastAPI + OpenCV | 轻量级推理服务 |

### 1.4 MVP 适用边界（V1 约束）

二值占用模型成立的前提，需提前与业务方对齐：

- **每槽位单件、单层摆放**：槽位状态只有"有/无"两种，不支持同槽位多件计数。
- **不支持纵深堆叠**：前排商品被取走、后排补位的摆放方式会导致漏报 pick，V1 明确不支持；升级路径见 6.6。
- 跟踪器（ByteTrack）在 MVP 阶段不启用，见 6.4。

---

## 二、系统架构

### 2.1 整体架构图

```
+--------------+     +----------------------------------------------+     +--------------+
|  摄像头 / RTSP  |---->|            推理服务 (Pipeline)                 |---->|  业务后端      |
|  IP Camera   |     |                                              |     |  (REST/gRPC) |
+--------------+     |  +--------+  +--------+  +--------+          |     +--------------+
                     |  | 检测器  |->| 跟踪器  |->| 事件判定 |          |
                     |  |Detector|  |Tracker |  | Event  |          |
                     |  |(YOLO)  |  |(ByteT) |  |Engine  |          |
                     |  +--------+  +--------+  +---+----+          |
                     |                             |               |
                     |                        +----v----+          |
                     |                        | 状态快照  |          |
                     |                        | Shelf   |          |
                     |                        | State   |          |
                     |                        +---------+          |
                     +----------------------------------------------+
```

### 2.2 模块职责

#### 2.2.1 视频输入层 (`pipeline/io`)

- 支持 RTSP 流、USB 摄像头、本地视频文件三种输入源。
- 帧采样控制：可配置 FPS（默认 5-10 fps，货架场景变化低频，足够用）。
- 帧预处理：resize、归一化、颜色空间转换。

#### 2.2.2 检测模块 (`pipeline/detector`)

- 加载 YOLO 模型（PyTorch -> ONNX 导出，ONNX Runtime 推理）。
- 输出每帧的检测框列表：`[x1, y1, x2, y2, class_id, confidence]`。
- 支持 ROI 裁剪：只推理货架区域，减少无效计算。
- 类别映射：输出 class_id -> SKU 名称 / SKU ID 的映射。

#### 2.2.3 跟踪模块 (`pipeline/tracker`)

- 为每个检测框分配稳定 track_id，跨越连续帧。
- 处理短暂遮挡（顾客手部遮挡商品）和 re-id。
- 输出：`[x1, y1, x2, y2, track_id, class_id, confidence]`。
- **MVP 阶段不启用**：槽位状态机不依赖 track_id（见 6.4），管线默认将检测结果直接送入 ShelfState。保留模块接口，后续用于"商品离开货架区域"的 pick 二次确认。

#### 2.2.4 货架状态建模 (`pipeline/shelf_state`)

- 货架配置：预定义槽位（slot）的 RoI 多边形区域，每个槽位对应一个 SKU 类型的期望位置。
- 每个槽位维护状态：
  - `occupied`: 该槽位当前是否有商品。
  - `matched_track_id`: 当前占据该槽位的跟踪 ID。
  - `last_seen`: 最后检测到商品的时间戳。
- 状态更新策略：
  - 每帧用检测框与槽位 RoI 做 IoU 匹配。
  - 帧间平滑（去抖窗口）避免 flicker；跟踪器启用时可结合 track_id 做跨帧关联。
- **遮挡门控（occlusion gating）**：
  - 利用检测器的 person 类（YOLO COCO 预训练自带，无需额外模型）检测顾客身体/手部。
  - 当 person 检测框与槽位 RoI 重叠超过阈值时，**冻结该槽位状态机**，期间不更新占用状态、不产生事件。
  - 遮挡解除后恢复判定；设置冻结超时（默认 30 秒）防止状态机永久卡死。

#### 2.2.5 事件判定引擎 (`pipeline/event_engine`)

- 基于帧间槽位状态变化判定事件：
  - **pick**: `occupied` 从 True -> False，且持续 N 帧（去抖）。
  - **place**: `occupied` 从 False -> True，且持续 N 帧，且检测类别与槽位期望 SKU 一致。
  - **misplace**: place 条件成立但检测类别与槽位期望 SKU 不一致，事件附带 `expected_sku` 和 `detected_sku`，由业务后端决策。
- 输出事件：`{event_id, timestamp, event_type, sku_id, slot_id, confidence, image_crop}`。
- `confidence` 定义：去抖窗口内相关检测框置信度的均值，反映本次状态判定的可信度。
- 去抖窗口：默认 3-5 帧，避免检测抖动产生伪事件。
- **补货模式**：通过 API 开关进入（见 8.2），期间抑制事件上报（店员一次补货多件会产生事件风暴），退出时以当前槽位状态为新基线。

#### 2.2.6 输出层 (`pipeline/output`)

- WebSocket / REST API 推送事件到业务后端。
- 可选本地 SQLite 存储事件日志。
- 可选帧快照保存（用于事后审计和模型迭代）。

---

## 三、目录结构

```
shelf/
├── config/                    # 配置文件
│   ├── default.yaml           # 默认全局配置
│   ├── shelf_layout.yaml      # 货架槽位 RoI 配置
│   └── model_mapping.yaml     # class_id -> SKU 映射
├── pipeline/                  # 核心推理管线
│   ├── __init__.py
│   ├── io/                    # 视频输入
│   │   ├── __init__.py
│   │   ├── source.py          # VideoSource 抽象 + RTSP/USB/File 实现
│   │   └── preprocess.py      # 帧预处理
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py   # YOLO 检测器封装（ONNX 推理）
│   │   └── roi.py             # ROI 裁剪逻辑
│   ├── tracker/
│   │   ├── __init__.py
│   │   └── byte_tracker.py    # ByteTrack 封装
│   ├── shelf_state.py         # 货架槽位状态管理
│   ├── event_engine.py        # pick/place 事件判定
│   └── output.py              # 事件输出（WS/REST/DB）
├── server/                    # 服务入口
│   ├── __init__.py
│   ├── app.py                 # FastAPI 应用
│   └── routes.py              # API 路由
├── tools/                     # 辅助工具
│   ├── label_tool.py          # 货架 RoI 标注工具
│   ├── dataset_builder.py     # 数据集构建脚本
│   └── model_export.py        # PyTorch -> ONNX 导出
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_shelf_state.py
│   └── test_event_engine.py
├── models/                    # 模型文件（.onnx）
├── data/                      # 数据目录（gitignore）
│   ├── raw/                   # 原始采集视频/图片
│   ├── annotated/             # 标注数据
│   └── snapshots/             # 事件快照
├── files/                     # 文档
│   ├── design.md
│   └── baseinfo.md
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 四、数据流

### 4.1 单帧处理流程

```
Frame(t) --> [ROI Crop] --> [YOLO Detector] --> detections[] 
                                                      |
                                          +-----------v-----------+
                                          |   ByteTrack Tracker   |
                                          |  detections[] + tracks |
                                          +-----------+-----------+
                                                      |
                            +-------------------------v-------------------------+
                            |                 ShelfState                         |
                            |  tracks[] -> slot IoU 匹配 -> 更新每个 slot 的占用状态 |
                            +-------------------------+-------------------------+
                                                      |
                                                      v
                                          +-----------------------+
                                          |     EventEngine        |
                                          |  slot 状态变化 + 去抖   |
                                          |  -> pick / place 事件   |
                                          +-----------------------+
```

### 4.2 事件判定状态机

```
         +---------+
    +--->| OCCUPIED |---+
    |    +---------+   |
    |  商品消失 > N 帧  |  商品出现 > N 帧
    |                  |
    |   +----------+   |
    +---|  EMPTY   |<--+
        +----------+

   OCCUPIED -> EMPTY  : 触发 pick 事件
   EMPTY -> OCCUPIED  : 触发 place 事件
```

- **去抖 N**：可配置，建议初始值 3-5 帧（对应 0.3-1.0 秒 @ 5fps）。
- **中间态**：`OCCUPIED -> MAYBE_EMPTY -> EMPTY`，`MAYBE_EMPTY` 为去抖缓冲区，不产生事件。
- **遮挡冻结态（FROZEN）**：person 检测框与槽位 RoI 重叠时进入，冻结期间不做状态转移、不产生事件，遮挡解除后回到原状态继续判定（见 2.2.4 / 6.5）。
- **售罄态（OUT_OF_STOCK）**：`EMPTY` 持续超过配置时长（默认 300 秒）后进入，用于区分"刚被拿空"和"长期缺货"；检测到 place 后回到 OCCUPIED。该状态通过 `/api/shelf/state` 暴露，可供运营做补货提醒。

---

## 五、开发阶段

### Phase 1：基础环境搭建（2-3 天）

- [ ] 确认 conda 环境 `yolo`，补充依赖清单。
- [ ] 安装 YOLOv8/v11、ONNX Runtime、OpenCV、FastAPI。
- [ ] 搭建项目骨架（目录结构、配置加载、logging）。
- [ ] 编写模型导出脚本：YOLO -> ONNX。

### Phase 2：核心管线开发（5-7 天）

- [ ] 实现 `pipeline/io/source.py`：RTSP/USB/File 视频源。
- [ ] 实现 `pipeline/detector/yolo_detector.py`：ONNX 推理封装。
- [ ] （可选，MVP 不启用）实现 `pipeline/tracker/byte_tracker.py`：ByteTrack 跟踪。
- [ ] 实现 `pipeline/shelf_state.py`：槽位状态管理（含遮挡门控，见 6.5）。
- [ ] 实现 `pipeline/event_engine.py`：pick/place/misplace 事件判定。
- [ ] 实现 `pipeline/output.py`：事件输出。

### Phase 3：标注工具与离线测试（3-4 天）

- [ ] 实现 `tools/label_tool.py`：货架 RoI 标注工具（基于 OpenCV GUI）。
- [ ] 准备少量测试视频，标注 ground truth 事件。
- [ ] 编写单元测试覆盖各模块。
- [ ] 端到端离线测试，计算 pick/place 事件准确率。

### Phase 4：服务化与部署（3-4 天）

- [ ] 实现 `server/app.py`：FastAPI + WebSocket 实时推送。
- [ ] 实现多路摄像头并发推理。
- [ ] 性能优化：帧率控制、内存管理、ONNX 量化（INT8）。
- [ ] Docker 化部署。

### Phase 5：联调与上线（3-5 天）

- [ ] 与业务后端对接事件格式。
- [ ] 现场摄像头接入调试。
- [ ] 异常场景处理：遮挡、光照变化、商品堆叠。
- [ ] 监控和告警。

---

## 六、关键设计决策

### 6.1 为什么用槽位 RoI 而不是纯检测框匹配

纯检测框做 pick/place 判定需要每帧对所有商品做重识别，容易受遮挡、光照、手部交互影响。而将货架划为固定槽位 RoI，每个槽位只追踪"有/没有商品"这个二值状态，大幅简化问题，且槽位位置固定，鲁棒性更强。

### 6.2 去抖机制的必要性

YOLO 检测存在帧间 flicker（同一目标相邻帧的检测置信度波动），不加去抖会产生大量伪 pick/place 事件。引入 N 帧去抖窗口后，只有持续的状态变化才会触发事件。

### 6.3 ONNX 导出而非 PyTorch 直接推理

- ONNX Runtime 推理速度优于 PyTorch（尤其是 CPU 场景）。
- 不依赖 PyTorch 全家桶，部署镜像更小。
- 支持 INT8 量化，边缘设备（Jetson / Intel NUC）友好。

### 6.4 MVP 阶段不启用跟踪器

槽位状态机基于"检测框 ↔ 槽位 IoU 匹配 + 去抖"即可工作，不依赖 track_id；货架商品基本静止，跟踪器解决的问题（跨帧 ID 关联）并非事件判定的必要条件。MVP 砍掉跟踪器可减少管线复杂度和故障面。

后续若需要"商品被拿起后离开货架区域"的 pick 二次确认，再启用跟踪器。届时选 ByteTrack 而非 DeepSORT：不需要额外的 ReID 特征提取网络，推理更快，对相对静态的货架场景性能足够。

### 6.5 遮挡门控的必要性

去抖窗口（3-5 帧，约 1 秒）只能过滤检测 flicker，防不了持续遮挡：顾客在货架前挑选 10 秒，身体挡住槽位会触发伪 pick，走开后又触发伪 place——成对的伪事件对自动结算是致命的。引入遮挡门控后，人/手与槽位重叠期间状态机冻结，只在视野干净时做判定，以极低的实现成本（复用检测器的 person 类，一条门控规则）消除最高频的误报来源。

### 6.6 数量计数的升级路径

V1 的二值占用模型不支持同槽位多件堆叠（见 1.4）。后续升级方向（按成本排序）：

1. **同槽位检测框计数**：在槽位 RoI 内统计同类检测框数量，适用于横向并排、无纵深遮挡的摆放。
2. **纵深堆叠**：纯单目视觉难以可靠解决，需融合深度相机或重量传感器，事件由"视觉 + 重量变化"联合判定。

---

## 七、配置示例

### default.yaml

```yaml
pipeline:
  input:
    type: rtsp              # rtsp | usb | file
    source: "rtsp://192.168.1.100:554/stream"
    fps: 5
    width: 1920
    height: 1080

  detector:
    model_path: "models/yolo11n.onnx"
    conf_threshold: 0.5
    iou_threshold: 0.45
    roi: [0, 0.2, 1.0, 0.9]   # 货架区域 [x1_ratio, y1_ratio, x2_ratio, y2_ratio]

  tracker:
    enabled: false              # MVP 不启用，见 6.4
    track_thresh: 0.5
    match_thresh: 0.8
    track_buffer: 30

  occlusion:                    # 遮挡门控，见 6.5
    enabled: true
    person_class_id: 0          # COCO person 类
    freeze_iou: 0.2             # person 框与槽位重叠超过该值则冻结状态机
    max_freeze_seconds: 30      # 冻结超时保护

  event:
    debounce_frames: 5          # 去抖帧数
    min_iou_for_slot: 0.3       # 检测框与槽位匹配的 IoU 阈值
    out_of_stock_seconds: 300   # EMPTY 持续超过该时长进入 OUT_OF_STOCK

server:
  host: "0.0.0.0"
  port: 8000
  ws_path: "/ws/events"

logging:
  level: INFO
  file: "logs/pipeline.log"
```

---

## 八、接口定义

### 8.1 事件输出格式

```json
{
  "event_id": "evt_20260612_001",
  "timestamp": "2026-06-12T10:23:45.123+08:00",
  "camera_id": "cam_01",
  "event_type": "pick",
  "sku_id": "cola_330ml",
  "slot_id": "A3",
  "confidence": 0.92,
  "snapshot_path": "data/snapshots/evt_20260612_001.jpg"
}
```

- `confidence`：去抖窗口内相关检测框置信度的均值（定义见 2.2.5）。
- misplace 事件额外携带期望与实际 SKU，便于业务后端决策：

```json
{
  "event_type": "misplace",
  "slot_id": "A3",
  "expected_sku": "sprite_330ml",
  "detected_sku": "cola_330ml"
}
```

### 8.2 REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | 健康检查 |
| GET | `/api/shelf/state` | 当前货架槽位状态快照 |
| GET | `/api/events?since=<ts>` | 查询历史事件 |
| GET | `/api/snapshot/<event_id>` | 获取事件截图 |
| POST | `/api/mode/restock` | 进入/退出补货模式（body: `{"enabled": true}`），期间抑制事件上报 |

### 8.3 WebSocket

- 路径：`/ws/events?camera_id=<id>`
- 实时推送 pick/place 事件，格式同上。

---

## 九、风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 光照变化导致检测不稳定 | 数据增强（亮度/对比度），动态阈值调整 |
| 商品外观极相似（同品牌不同口味） | 细粒度分类、槽位位置先验 |
| 多人同时操作 | 槽位级独立状态机 + 遮挡门控（6.5） |
| 手部/身体持续遮挡商品 | 遮挡门控：遮挡期间冻结槽位状态机（6.5） |
| 商品纵深堆叠导致漏报 pick | V1 限定每槽单件单层（1.4），升级路径见 6.6 |
| 新 SKU 上线需要重新训练 | 数据闭环流程（第十一节），支持热更新模型 |

---

## 十、评估与验收指标

- **指标定义**：pick / place / misplace 分别统计 precision 和 recall。预测事件与 ground truth 事件的匹配采用时间对齐容差 **±2 秒**：同 slot_id + 同 event_type + 时间窗内视为命中。
- **业务权重**：结算场景下漏报 pick 是直接资损、误报 pick 是客诉，阈值调优偏向需与业务方先行确认。初始建议：优先压低误报，pick precision ≥ 0.95、recall ≥ 0.90。
- **测试集要求**：覆盖五类场景的标注视频——正常取放、持续遮挡、多人操作、补货、光照变化。

---

## 十一、数据闭环

模型质量依赖持续的数据回流：

1. **采集**：事件快照（`snapshot_path` 已预留）+ 低置信度帧定期落盘。
2. **抽审**：人工定期抽查事件快照，标记误报/漏报。
3. **回流**：误报/漏报样本进入标注队列，扩充训练集。
4. **重训与发布**：周期性重训检测模型，经离线指标（第十节）验证后通过 ONNX 热更新发布。

新 SKU 上线走同一流程：采集新 SKU 在架图像 -> 标注 -> 增量训练 -> 更新 `model_mapping.yaml` 与 `shelf_layout.yaml` -> 热更新模型。

---

## 十二、依赖清单

训练与部署依赖分离：`ultralytics` 会引入完整 PyTorch，只用于训练/导出环节，部署镜像不安装（与 6.3 的部署目标一致）。

```
# requirements-deploy.txt（推理服务）
onnxruntime>=1.16.0         # ONNX 推理
opencv-python>=4.8.0        # 视频处理
numpy>=1.24.0               # 数值计算
PyYAML>=6.0                 # 配置解析
fastapi>=0.100.0            # Web 服务
uvicorn>=0.23.0             # ASGI 服务器
websockets>=12.0            # WebSocket 推送
loguru>=0.7.0               # 日志
```

```
# requirements-train.txt（训练/导出/测试，开发机使用）
ultralytics>=8.0.0          # YOLO 训练和导出
pytest>=7.4.0               # 测试
boxmot>=10.0.0              # ByteTrack 实现（可选，MVP 不启用，见 6.4）
```

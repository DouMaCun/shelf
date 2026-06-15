# Python 小白设计思想指引

> 这份文档面向 Python 初学者，解释 shelf 项目里用到的编程模式和设计思想。  
> 每一节都先说"是什么"，再用生活类比帮你建立直觉，最后看代码。  
> 建议配合项目源码一起阅读。

---

## 目录

1. [管线（Pipeline）](#一管线pipeline)
2. [抽象基类（ABC）](#二抽象基类abc)
3. [工厂函数](#三工厂函数)
4. [数据类（dataclass）](#四数据类dataclass)
5. [状态机 + 去抖](#五状态机--去抖)
6. [回调函数（Callback）](#六回调函数callback)
7. [后台线程](#七后台线程)
8. [策略模式（Strategy Pattern）](#八策略模式strategy-pattern)
9. [整体结构总览](#九整体结构总览)

---

## 一、管线（Pipeline）

### 是什么

把一个复杂任务拆成**一系列顺序步骤**，每步只做一件事，前一步的输出是后一步的输入。就像工厂流水线：原材料依次经过各个工位，最终变成成品。

### 生活类比

咖啡机制作咖啡：
```
磨豆 → 压粉 → 萃取 → 加奶 → 出杯
```
每步都有明确输入和输出，中间任何一步出问题都很容易定位。

### 项目中的管线

`server/app.py` 的 `run_pipeline_loop()` 函数就是这条流水线：

```python
# 每帧都经历以下步骤（状态不同，采样频率不同）
frame = video_source.read()                          # 第1步：取一帧画面
detections = detector.detect(frame)                  # 第2步：手臂检测（YOLO）
has_person = any(d[5] == person_class_id for d in detections)
new_state = interaction_monitor.update(frame, has_person)  # 第3步：推进状态机

# 第4步：状态变为 CLASSIFYING 时触发识别
if new_state == CLASSIFYING:
    best_frame = interaction_monitor.get_best_frame()   # 选最清晰帧
    sku_id, conf = _identify_item(best_frame)           # YOLO视觉分类
    event = event_engine.emit_pick(sku_id, conf, best_frame)
    await event_output.broadcast_async(event, ws_clients)  # 第5步：推送
    interaction_monitor.complete_classification()
```

数据流向：
```
摄像头帧
  ↓ (ndarray)
手臂检测结果  [x1,y1,x2,y2,conf,class_id]
  ↓ (has_person: bool)
状态机推进  IDLE → ACTIVE → CLASSIFYING → COOLDOWN
  ↓ (最清晰帧: ndarray)
YOLO 视觉分类  → (sku_id, confidence)
  ↓
ShelfEvent(pick, sku_id, confidence)
  ↓
WebSocket 推送 / 截图保存 / 数据库写入
```

### 为什么这样设计

- **隔离**：每步独立，可以单独测试（`tests/` 里每个模块都有自己的测试文件）
- **可替换**：想把 YOLO 换成别的模型，只改第3步，其他步骤不动
- **好调试**：出了问题，打印中间结果就能找到是哪步坏了

---

## 二、抽象基类（ABC）

### 是什么

**抽象基类（Abstract Base Class）**定义一套接口"规范"，强制要求子类必须实现某些方法。它本身不能被直接使用，只能被继承。

### 生活类比

USB 接口标准：
- 标准规定了接口形状、电压、数据协议（这就是抽象基类）
- 你的鼠标、U盘、手机线都符合这个标准（这就是子类）
- 电脑只认 USB 接口，不管你插的是哪个厂家的设备（这就是"面向接口编程"）

### 项目中的抽象基类

`pipeline/io/source.py` 的 `VideoSource`：

```python
from abc import ABC, abstractmethod

class VideoSource(ABC):          # ABC 表示这是抽象基类
    
    # 子类"必须"实现这两个方法，否则实例化时会报错
    @abstractmethod
    def _open(self): ...         # 怎么打开这个视频源？
    
    @abstractmethod
    def _read_frame(self): ...   # 怎么读一帧？
    
    # 下面这些方法所有子类共享，写一次就够了
    def start(self): ...         # 启动后台线程（通用逻辑）
    def stop(self): ...          # 停止并释放资源（通用逻辑）
    def read(self): ...          # 从缓冲区取帧（通用逻辑）
```

三个子类**只需填写差异部分**：

```python
class FileSource(VideoSource):
    def _open(self):
        return cv2.VideoCapture("demo.mp4")   # 打开文件
    
    def _read_frame(self):
        ret, frame = self._capture.read()
        return frame if ret else None          # 文件读完返回 None

class RTSPSource(VideoSource):
    def _open(self):
        return cv2.VideoCapture("rtsp://...")  # 打开网络流
    
    def _read_frame(self):
        ret, frame = self._capture.read()
        if not ret:
            # 网络断了，等2秒重连（这是 RTSP 特有的逻辑）
            time.sleep(2)
            self._capture = self._open()
            ret, frame = self._capture.read()
        return frame if ret else None
```

管线代码永远只写：
```python
src = create_source(config)   # 不管是文件还是摄像头
src.start()
frame = src.read()            # 统一接口，无需关心底层实现
```

### 为什么这样设计

- 公共逻辑（线程管理、帧率控制、缓冲队列）只写一次，在基类里
- 新增一种输入源（比如 HTTP 视频流）只需新建子类，实现两个方法
- 降低了新人上手成本：看基类接口就知道该实现什么

---

## 三、工厂函数

### 是什么

**工厂函数**是一个"创建对象的函数"。调用方不需要知道具体类名，只告诉工厂"我要什么类型"，工厂负责返回正确的对象。

### 生活类比

奶茶店点单：
- 你说"要一杯珍珠奶茶"（输入类型名称）
- 店员去后台做（工厂函数内部逻辑）
- 你拿到杯子（返回对象）
- 你不需要知道奶茶怎么做的

### 项目中的工厂函数

`pipeline/io/source.py` 的 `create_source()`：

```python
def create_source(config: dict) -> VideoSource:
    source_type = config["type"]   # "file" / "usb" / "rtsp"
    
    if source_type == "file":
        return FileSource(...)     # 返回文件源对象
    elif source_type == "usb":
        return USBSource(...)      # 返回摄像头对象
    elif source_type == "rtsp":
        return RTSPSource(...)     # 返回网络流对象
    else:
        raise ValueError(f"不支持的类型: {source_type}")
```

调用方（`server/app.py`）只写一行：
```python
video_source = create_source(pipeline_cfg["input"])
```

想从文件换成 RTSP 摄像头？只改 `config/default.yaml`：
```yaml
# 改这一行，代码零改动
type: rtsp   # 之前是 file
```

### 为什么这样设计

把"选择创建哪个类"的逻辑集中到一个地方，避免在各处散落 `if/elif`。

---

## 四、数据类（dataclass）

### 是什么

`@dataclass` 是 Python 的语法糖，自动帮你生成 `__init__`、`__repr__` 等样板代码，让你专注于"这个数据结构有哪些字段"。

### 对比一下

不用 dataclass 的写法（冗余）：
```python
class ShelfEvent:
    def __init__(self, event_id, timestamp, camera_id,
                 shelf_id, event_type, sku_id, confidence):
        self.event_id = event_id
        self.timestamp = timestamp
        self.camera_id = camera_id
        self.shelf_id = shelf_id
        self.event_type = event_type
        self.sku_id = sku_id
        self.confidence = confidence
```

用 dataclass 的写法（简洁）：
```python
@dataclass
class ShelfEvent:
    event_id: str
    timestamp: str
    camera_id: str
    shelf_id: str
    event_type: str       # "pick" | "place"
    sku_id: str
    confidence: float
    snapshot_path: str = ""      # = 号后面是默认值
    frame: np.ndarray | None = None
```

效果完全相同，但省去了大量 `self.xxx = xxx` 样板代码。

### 项目中的数据类

**`ShelfEvent`**（`pipeline/event_engine.py`）— 一次取走商品的事件：
```python
@dataclass
class ShelfEvent:
    event_id: str         # 唯一ID，如 "evt_a3f2b1c0d4e5"
    timestamp: str        # ISO 8601 时间戳
    camera_id: str        # 产生事件的摄像头
    shelf_id: str         # 产生事件的货架
    event_type: str       # "pick"（取走）或 "place"（放回）
    sku_id: str           # 识别到的商品 SKU
    confidence: float     # 识别置信度
    snapshot_path: str = ""            # 截图路径（可选）
    frame: np.ndarray | None = None    # 原始帧图像（不序列化）
```

### 为什么这样设计

数据结构清晰可读，字段类型一目了然，代码量减少一半。

---

## 五、状态机

### 是什么

**状态机**：系统在有限个状态之间转换，每次转换由"触发条件"决定。状态之外什么都不做，避免在不需要的时候消耗资源。

### 生活类比

超市自动门：有人靠近（传感器触发）→ 开门（ACTIVE）→ 人走完（关门中 CLOSING）→ 完全关闭（IDLE）。不管外面风吹草动，只有真正有人靠近才响应。

### 项目中的交互状态机

`pipeline/interaction_monitor.py` 的 `InteractionMonitor`，管理四个状态：

```
IDLE ──手臂进入视野──► ACTIVE ──手臂离开──► CLASSIFYING ──识别完成──► COOLDOWN ──► IDLE
                         │                      │
                    持续缓冲帧              取最清晰帧执行识别
```

```python
def update(self, frame, has_person):
    if self._state == IDLE:
        if has_person:
            self._state = ACTIVE          # 手臂出现，切高频采样
            self._frame_buffer.clear()

    elif self._state == ACTIVE:
        self._frame_buffer.append(frame)  # 持续缓冲
        if not has_person:
            self._state = CLASSIFYING     # 手臂消失，准备识别

    elif self._state == COOLDOWN:
        if time.time() >= self._cooldown_until:
            self._state = IDLE            # 冷却结束
    
    return self._state
```

各状态说明：

| 状态 | 含义 | 采样频率 |
|---|---|---|
| **IDLE** | 等待有人伸手 | 低频（5fps） |
| **ACTIVE** | 手臂在视野内，缓冲帧 | 高频（10fps） |
| **CLASSIFYING** | 手臂离开，执行识别 | 暂停采样 |
| **COOLDOWN** | 事件已发出，静默 N 秒防重复 | 低频 |

### "最清晰帧"是怎么选的

商品被举起穿过摄像头正下方时，离镜头最近，图像最清晰。用 **Laplacian 方差**衡量：

```python
def get_best_frame(self):
    best_score = -1
    for f in self._frame_buffer:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()  # 方差越大越清晰
        if score > best_score:
            best_score, best_frame = score, f
    return best_frame
```

### 为什么这样设计

状态机让系统在 IDLE 时只跑轻量手臂检测（1 个 YOLO 推理/帧），只在 CLASSIFYING 时才执行耗时的 YOLO 视觉分类，大幅降低平均计算量。

---

## 六、回调函数（Callback）

### 是什么

**回调函数**是你"提前注册"的一个函数，某个事件发生时系统自动调用它。你不用主动查询，等通知就行。

### 生活类比

快递短信通知：
- 你填了手机号（注册回调）
- 快递到了系统自动发短信给你（事件触发，调用回调）
- 你不需要每隔10分钟打电话问快递到了没有

### 项目中的回调

`pipeline/event_engine.py` 的 `EventEngine`：

```python
# 初始化时传入一个函数
engine = EventEngine(on_event=my_handler)

# 每次产生事件，引擎自动调用这个函数
def my_handler(event: ShelfEvent):
    print(f"事件: {event.event_type} | sku={event.sku_id} | conf={event.confidence:.2f}")
    # 可以在这里发送HTTP请求、写数据库、发报警...
```

引擎内部的调用：
```python
# EventEngine.emit_pick() 内部
event = ShelfEvent(...)
self._events.append(event)
if self._on_event:
    self._on_event(event)   # 自动回调，不需要外部主动查询
```

不传回调也可以，之后手动查询历史事件：
```python
engine = EventEngine()   # 不传 on_event
# ...运行一段时间后...
events = engine.get_events(since=timestamp)   # 主动拉取
```

### 为什么这样设计

解耦生产者（EventEngine 产生事件）和消费者（谁来处理事件）。引擎不需要知道事件要怎么处理，只管产生并通知。

---

## 七、后台线程

### 是什么

**后台线程（background thread）**是在主程序之外同时运行的"副线程"。主线程和副线程并发执行，互不等待。

### 为什么需要后台线程

读取摄像头帧是个慢操作（需要等摄像头响应）。如果在主线程里读，整个程序每读一帧就得停下来等一下，推理管线就会卡顿。

解决方案：让一个专门的后台线程负责读帧，主线程想用的时候直接去"取件箱"（缓冲队列）拿，不需要等待。

### 生活类比

快递驿站：
- 快递员（后台线程）一直在收货、上架
- 你（主线程）想取快递了，直接去驿站拿
- 两者互不等待

### 项目中的后台线程

`pipeline/io/source.py` 的 `VideoSource`：

```python
def start(self):
    # 启动后台线程，daemon=True 表示主程序退出时它自动停
    self._thread = threading.Thread(
        target=self._read_loop,   # 后台线程执行的函数
        daemon=True
    )
    self._thread.start()

def _read_loop(self):
    """后台线程主循环：不断读帧，放入缓冲队列。"""
    while self._running:
        frame = self._read_frame()          # 读一帧（可能需要等待）
        self.buffer.append(frame)           # 放入右侧
        time.sleep(0.001)                   # 短暂休息，避免CPU空转

def read(self):
    """主线程调用：从缓冲队列取帧（不等待）。"""
    try:
        return self.buffer.popleft()        # 从左侧取，没有就返回 None
    except IndexError:
        return None                          # 缓冲区空了，本帧跳过
```

缓冲队列用 `deque(maxlen=30)`：
- 后台线程从右边放帧
- 主线程从左边取帧
- 队列满了自动丢弃最旧的帧（要的是"实时"，不是"历史"）

```
后台线程 →  [帧27][帧28][帧29][帧30]  → 主线程取走
              最旧帧                   最新帧
```

### 为什么这样设计

后台线程和主管线并发跑，两者互不阻塞。视频读取速度和推理速度解耦，整个管线运行更平稳。

---

## 八、策略模式（Strategy Pattern）

### 是什么

**策略模式**：把"做什么"和"怎么做"分开。同一个功能有多种实现方式（策略），调用方不需要改代码，只通过配置来切换策略。

### 生活类比

导航 App 的出行方式：你输入目的地后，可以选"开车""地铁""骑行"——目标（到达目的地）不变，策略（交通方式）可以随时切换。App 的其余逻辑（显示地图、播报距离）完全不变。

### 项目中的策略模式：检测器后端

这个项目在部署到 RK3588 边缘设备时，推理后端需要从 ONNX Runtime（x86 CPU）切换到 RKNN（RK3588 NPU）。  
但管线其余部分（取帧、跟踪、状态判定）完全不需要改动。

**两个检测器类，接口完全相同：**

```python
# pipeline/detector/yolo_detector.py（x86 开发机）
class YOLODetector:
    def load(self): ...                        # 加载 .onnx 模型
    def detect(frame) -> ndarray(N, 6): ...   # ONNX Runtime 推理

# pipeline/detector/rknn_detector.py（RK3588 设备）
class RKNNDetector:
    def load(self): ...                        # 加载 .rknn 模型，初始化 NPU
    def detect(frame) -> ndarray(N, 6): ...   # RKNN NPU 推理
```

两个类的 `detect()` 输出格式完全一致：`ndarray(N, 6)` = `[x1, y1, x2, y2, conf, class_id]`。管线代码调用 `detector.detect(frame)` 时根本不知道底层是哪个引擎。

**`server/app.py` 里的切换逻辑（只有这几行）：**

```python
backend = det_cfg.get('backend', 'onnx')   # 读配置
if backend == 'rknn':
    detector = RKNNDetector(**det_kwargs)   # 选 NPU 策略
else:
    detector = YOLODetector(**det_kwargs)   # 选 ONNX 策略
detector.load()
# 之后管线代码完全不变
```

**切换只改一行 YAML：**

```yaml
# config/default.yaml
detector:
  backend: onnx   # 改成 rknn 即切换到 RK3588 NPU，代码零改动
  model_path: "models/yolo11n.onnx"   # rknn 时改为 .rknn 路径
```

### 模型转换流程

RKNN 不能直接用 `.onnx` 文件，需要在开发机上转换：

```
开发机（x86）                          RK3588 设备
─────────────────────────────────────────────────────
yolo11n.pt
  ↓ tools/model_export.py
yolo11n.onnx
  ↓ tools/model_export_rknn.py         ─── 拷贝 .rknn 到设备 ───►  yolo11n.rknn
yolo11n.rknn                                                          ↓ RKNNDetector.load()
                                                                      NPU 推理
```

### 为什么这样设计

两个检测器保持**相同接口**是关键。这意味着：
- 新增 OpenVINO、TensorRT 等后端，只需新增一个类、加一个 `elif`，其余代码不动
- 单元测试可以用 `YOLODetector` 跑，部署时换 `RKNNDetector`，行为一致

---

## 九、整体结构总览

### 目录与管线的对应关系

```
项目目录                        对应管线环节
────────────────────────────────────────────────────────
pipeline/io/                    第1步：取帧
  source.py                       VideoSource（抽象基类 + 3种实现）
  preprocess.py                   帧预处理工具函数

pipeline/detector/              第2步：检测（手臂 / 商品，两个实例）
  yolo_detector.py                YOLODetector（ONNX Runtime）
  rknn_detector.py                RKNNDetector（RK3588 NPU）
  roi.py                          ROI 区域裁剪

pipeline/interaction_monitor.py 第3步：交互状态机
                                  IDLE/ACTIVE/CLASSIFYING/COOLDOWN
                                  帧缓冲 + 最清晰帧选择

pipeline/identifier/            第4步：商品识别
  item_classifier.py              YOLO视觉分类 → (sku_id, confidence)

pipeline/event_engine.py        第5步：生成事件
                                  emit_pick() → ShelfEvent

pipeline/output.py              第6步：输出
                                  WebSocket推送 / 截图 / SQLite

server/app.py                   总控制器（FastAPI + asyncio）

config/
  default.yaml                    全局配置（含 interaction / item_classifier 节）
  model_mapping.yaml              class_id → SKU名称

tools/
  model_export.py                 YOLO .pt → ONNX
  model_export_rknn.py            ONNX → RKNN（RK3588 部署前运行）
```

### 各模块之间的数据格式约定

| 步骤 | 输出格式 | 说明 |
|---|---|---|
| VideoSource.read() | `ndarray (H, W, 3)` | BGR 图像，uint8 |
| YOLODetector.detect() | `ndarray (N, 6)` | [x1,y1,x2,y2,conf,class_id] |
| InteractionMonitor.update() | `InteractionState` | IDLE/ACTIVE/CLASSIFYING/COOLDOWN |
| InteractionMonitor.get_best_frame() | `ndarray (H, W, 3)` | 最清晰帧，用于识别 |
| ItemClassifier.classify() | `(str \| None, float)` | (sku_id, confidence) |
| EventEngine.emit_pick() | `ShelfEvent` | 单个 pick 事件 |

### 设计原则小结

| 原则 | 在项目中的体现 |
|---|---|
| **单一职责** | 每个模块只做一件事，不越界 |
| **面向接口** | 管线只认统一接口，不关心具体实现（VideoSource、detect()） |
| **配置与代码分离** | 参数全在 YAML，代码不写死数值；切换推理后端只改一行配置 |
| **数据格式标准化** | 模块间用 ndarray/dict 传递，格式固定 |
| **可测试性** | 每个模块可独立传入测试数据，不依赖真实摄像头或 NPU |
| **可扩展性** | 新增输入源/推理后端只需新建类+一行 if-elif，其余代码不动 |

---

> 阅读顺序建议：先看 `server/app.py` 的 `run_pipeline_loop()` 理解全貌，再按管线顺序逐个模块深入。每个模块对应 `tests/` 下都有测试文件，看测试是理解一个模块最快的方式。
>
> 如需部署到 RK3588 边缘设备，参考 [`files/deploy_rk3588.md`](deploy_rk3588.md)。

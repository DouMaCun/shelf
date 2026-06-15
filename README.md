# Shelf Monitor

无人售货柜商品识别系统。摄像头装在柜门顶部向下俯拍，通过 YOLO 检测手臂进出动作，在商品被取走时自动识别 SKU 并输出 pick 事件。

## 工作原理

```
手臂进入视野
    ↓  YOLO 检测 person 类
交互状态机（IDLE → ACTIVE → CLASSIFYING → COOLDOWN）
    ↓  手臂离开后，从缓冲帧中选最清晰帧
YOLO 视觉分类
    ↓  class_id → sku_id（model_mapping.yaml）
pick 事件 → WebSocket 推送 / 截图 / SQLite
```

## 目录结构

```
shelf/
├── config/
│   ├── default.yaml          # 全局配置（输入源、检测器、状态机、输出）
│   └── model_mapping.yaml    # class_id → sku_id 映射
├── pipeline/
│   ├── io/source.py          # VideoSource（RTSP / USB / 文件）
│   ├── detector/
│   │   ├── yolo_detector.py  # ONNX Runtime 推理
│   │   └── rknn_detector.py  # RK3588 NPU 推理
│   ├── interaction_monitor.py# 4 状态机 + 帧缓冲 + 最清晰帧选择
│   ├── identifier/
│   │   └── item_classifier.py# YOLO 视觉分类 → (sku_id, confidence)
│   ├── event_engine.py       # ShelfEvent + emit_pick()
│   └── output.py             # WebSocket 广播 / 截图 / SQLite
├── server/app.py             # FastAPI 服务入口
├── tools/
│   ├── model_export.py       # YOLO .pt → ONNX
│   └── model_export_rknn.py  # ONNX → RKNN（RK3588 部署前运行）
├── tests/                    # 单元测试
├── models/                   # 模型文件（.onnx / .rknn）
├── data/                     # 截图、日志、SQLite 数据库
└── files/                    # 详细文档
    ├── design.md             # 系统设计文档
    ├── dev_guide.md          # 二次开发指引
    ├── deploy_rk3588.md      # RK3588 边缘部署指引
    └── beginner_guide.md     # 设计思想入门指引
```

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 启动服务（需先准备好模型和视频源，见 config/default.yaml）
python -m server.app -c config/default.yaml

# 仅启动 Web 服务，跳过推理管线（用于调试 API）
python -m server.app --no-pipeline

# 运行测试
pytest tests/
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，返回管线状态和推理性能指标 |
| GET | `/api/shelf/state` | 当前交互状态和最近 10 条事件 |
| GET | `/api/events?since=<ts>` | 查询历史事件 |
| GET | `/api/snapshot/{event_id}` | 获取事件截图 |
| WS  | `/ws/events` | 实时 pick 事件推送 |

pick 事件推送格式：

```json
{
  "event_id": "evt_a3f2b1c0d4e5",
  "timestamp": "2026-06-15T10:23:45",
  "camera_id": "cam_01",
  "event_type": "pick",
  "sku_id": "resistor_100ohm",
  "confidence": 0.87,
  "snapshot_path": "data/snapshots/evt_a3f2b1c0d4e5.jpg"
}
```

## 模型准备

```bash
# 1. 将 YOLO 模型导出为 ONNX（开发机）
python tools/model_export.py -m models/yolo11n.pt -o models/yolo11n.onnx

# 2. 转换为 RKNN（RK3588 部署，需 rknn-toolkit2）
python tools/model_export_rknn.py -i models/yolo11n.onnx -o models/yolo11n.rknn

# INT8 量化（速度约快 2x，需校准图片）
python tools/model_export_rknn.py -i models/yolo11n.onnx -o models/yolo11n_int8.rknn \
  --int8 --calib-dir data/raw/calib/
```

## 文档

- [二次开发指引](files/dev_guide.md) — 模块 API、扩展场景、接口说明
- [RK3588 边缘部署](files/deploy_rk3588.md) — NPU 部署步骤和性能参考
- [系统设计文档](files/design.md) — 架构决策和设计背景
- [设计思想入门](files/beginner_guide.md) — 面向 Python 初学者的设计模式说明

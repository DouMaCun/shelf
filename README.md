# Shelf Monitor

无人货架商品取放识别系统。基于 YOLO + ByteTrack 的视觉货架方案，实时检测顾客从货架上取走（pick）和放回（place）商品的行为。

## 目录结构

```
shelf/
├── config/          # 配置文件
├── pipeline/        # 核心推理管线
│   ├── io/          # 视频输入
│   ├── detector/    # YOLO 检测器
│   └── tracker/     # ByteTrack 跟踪器
├── server/          # FastAPI 服务
├── tools/           # 辅助工具
├── tests/           # 单元测试
├── models/          # ONNX 模型文件
└── data/            # 数据目录
```

## 快速开始

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行服务
python -m server.app -c config/default.yaml

# 运行测试
pytest tests/
```

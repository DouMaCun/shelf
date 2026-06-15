# RK3588 边缘部署指引

本文档说明如何将 shelf 系统部署到搭载 RK3588 芯片的边缘设备（如 Orange Pi 5、Rock 5B、Firefly ROC-RK3588S 等）。

---

## 为什么选 RK3588

| 资源 | 规格 | 对本项目的意义 |
|---|---|---|
| CPU | 4×Cortex-A76 + 4×Cortex-A55 | 可跑全套 Python 服务 |
| NPU | 3 核心，共 **6 TOPS** | YOLO 推理 5–30ms，满足 5fps 实时要求 |
| 内存 | 4/8/16 GB LPDDR5 | 足够同时跑推理服务和 FastAPI |
| 接口 | USB3 / MIPI CSI / PCIe | 直接接摄像头，无需采集卡 |

纯 CPU 推理（ONNX Runtime on ARM64）大约 200–500ms/帧，无法达到实时。接入 NPU 后可降至 **5–30ms/帧**。

---

## 工具链总览

```
开发机（x86 Ubuntu）               RK3588 设备（ARM64）
──────────────────────────────────────────────────────────
pip install ultralytics             pip install rknn-toolkit-lite2
pip install rknn-toolkit2           pip install -e ".[dev]"（项目依赖）

yolo11n.pt
  ↓ tools/model_export.py
yolo11n.onnx
  ↓ tools/model_export_rknn.py
yolo11n.rknn ──── scp ──────────► yolo11n.rknn
                                    ↓ RKNNDetector
                                    NPU 推理
```

两个工具链安装在不同机器，不能混用：
- `rknn-toolkit2`：仅支持 x86/Ubuntu，用于模型转换
- `rknn-toolkit-lite2`：仅支持 RK3588/RK3566 等 Rockchip 设备，用于设备侧推理

---

## Step 1：开发机准备模型

### 1.1 导出 ONNX（已有 .onnx 可跳过）

```bash
pip install ultralytics
python tools/model_export.py -m models/yolo11n.pt -o models/yolo11n.onnx
```

### 1.2 安装 RKNN 转换工具

```bash
pip install rknn-toolkit2
```

### 1.3 转换为 RKNN（FP16，推荐优先尝试）

```bash
python tools/model_export_rknn.py \
  -i models/yolo11n.onnx \
  -o models/yolo11n.rknn \
  --target rk3588
```

### 1.4 转换为 RKNN（INT8 量化，速度再快约 2x，精度损失 < 1%）

需要一批代表性图片做校准（50–200 张货架图）：

```bash
python tools/model_export_rknn.py \
  -i models/yolo11n.onnx \
  -o models/yolo11n_int8.rknn \
  --target rk3588 \
  --int8 \
  --calib-dir data/raw/calib/
```

---

## Step 2：拷贝文件到设备

```bash
# 拷贝模型文件
scp models/yolo11n.rknn user@<设备IP>:~/shelf/models/

# 或者整个项目目录（首次部署）
rsync -av --exclude='.git' --exclude='data/' \
  ~/shelf/ user@<设备IP>:~/shelf/
```

---

## Step 3：设备环境准备

SSH 登录设备后：

```bash
# 安装 rknn-toolkit-lite2（仅支持设备侧，x86 不能装）
pip install rknn-toolkit-lite2

# 安装项目依赖（不含 ultralytics，设备上不需要训练）
cd ~/shelf
pip install -e .
```

> **注意**：`requirements.txt` 中的 `ultralytics` 包含完整 PyTorch，仅用于训练和导出，
> 设备上可跳过不装（`pip install -e .` 会安装，但实际推理不会用到它）。

---

## Step 4：修改配置

编辑 `config/default.yaml`，改以下几处：

```yaml
pipeline:
  input:
    type: rtsp                                    # 或 usb
    source: "rtsp://192.168.1.100:554/stream"     # 改为实际摄像头地址

  detector:                                       # 手臂检测器（COCO 预训练）
    backend: rknn                                 # ← 从 onnx 改为 rknn
    model_path: "models/yolo11n.rknn"             # ← COCO 预训练模型的 rknn 路径
    conf_threshold: 0.5
    iou_threshold: 0.45

item_classifier:                                  # 商品识别器（自定义俯拍训练）
  backend: rknn
  model_path: "models/yolo11n_custom.rknn"        # ← 自定义模型的 rknn 路径（可选）
  model_mapping: "config/model_mapping.yaml"
```

> 若 `item_classifier.model_path` 留空，系统将复用手臂检测器做商品识别（仅适合调试）。

---

## Step 5：启动服务

```bash
# 验证服务能启动（不启动推理管线）
python -m server.app --no-pipeline

# 完整启动
python -m server.app -c config/default.yaml
```

访问健康检查接口，确认推理正常：

```bash
curl http://<设备IP>:8000/health
# 期望返回:
# {"status": "ok", "pipeline_running": true, "fps": 4.9, "inference_ms": 18.3}
```

---

## 性能参考

| 模型 | 精度 | NPU 推理耗时 | 备注 |
|---|---|---|---|
| yolo11n（FP16） | 最高 | ~20–30ms | 推荐生产环境首选 |
| yolo11n（INT8） | 略低（< 1% mAP 差距） | ~10–15ms | 校准集质量影响精度 |
| yolo11s（FP16） | 更高 | ~40–60ms | 5fps 下仍可接受 |

@5fps 推理帧间隔 200ms，NPU 推理耗时 30ms 以内均可满足实时要求。

---

## 常见问题

**Q：`ImportError: No module named 'rknnlite'`**  
A：`rknn-toolkit-lite2` 没装，或者在 x86 机器上运行了设备侧代码。参考 Step 3。

**Q：`RuntimeError: NPU 初始化失败，错误码: -1`**  
A：检查设备内核是否支持 RKNN NPU，运行 `ls /dev/rknpu*` 确认驱动存在。

**Q：推理结果正常但 FPS 很低（< 1fps）**  
A：检查 `config/default.yaml` 的 `backend` 是否确实改为 `rknn`；也可能是视频源读取成为瓶颈，检查 RTSP 连接延迟。

**Q：INT8 量化后精度明显下降**  
A：增加校准图片数量（建议 100 张以上），图片要覆盖真实部署场景（光照、角度、商品种类）。

**Q：想用 USB 摄像头代替 RTSP**  
A：修改 `config/default.yaml`：
```yaml
input:
  type: usb
  source: "0"   # 摄像头索引，通常是 0
```

---

## 相关文件

| 文件 | 说明 |
|---|---|
| `tools/model_export_rknn.py` | ONNX → RKNN 转换脚本（开发机运行） |
| `pipeline/detector/rknn_detector.py` | RK3588 NPU 推理类 |
| `pipeline/detector/yolo_detector.py` | ONNX Runtime 推理类（x86/ARM64 CPU） |
| `config/default.yaml` | 全局配置，`detector.backend` 控制推理后端 |

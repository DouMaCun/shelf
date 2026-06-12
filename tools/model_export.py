# -*- coding: utf-8 -*-
"""
PyTorch YOLO 模型 -> ONNX 导出脚本。

功能：
  将 Ultralytics 训练的 YOLO 模型（.pt）导出为 ONNX 格式（.onnx），
  供 YOLODetector (ONNX Runtime) 进行高效推理。

使用方式：
  python tools/model_export.py -m models/yolo11n.pt -o models/yolo11n.onnx

参数说明：
  -m, --model: 输入 PyTorch 模型路径（.pt）
  -o, --output: 输出 ONNX 模型路径（.onnx）
  --imgsz: 模型输入尺寸（默认 640）
  --opset: ONNX opset 版本（默认 12）
  --simplify: 是否 ONNX 图优化（默认开启）
  --int8: 是否 INT8 量化（默认关闭，需要校准数据集）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def export_to_onnx(model_path: str, output_path: str, imgsz: int = 640,
                   opset: int = 12, simplify: bool = True) -> None:
    """导出 YOLO 模型为 ONNX 格式。

    Args:
        model_path: PyTorch 模型路径（.pt）
        output_path: ONNX 输出路径
        imgsz: 模型输入尺寸
        opset: ONNX opset 版本
        simplify: 是否使用 onnx-simplifier 简化图结构
    """
    from ultralytics import YOLO

    # 加载 PyTorch 模型
    logger.info(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 导出 ONNX
    logger.info(f"导出 ONNX (imgsz={imgsz}, opset={opset})...")
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
    )

    # Ultralytics 默认导出到与 pt 同名的 onnx 文件
    # 如果指定了不同路径，则移动
    default_onnx = Path(model_path).with_suffix(".onnx")
    target_onnx = Path(output_path)
    if default_onnx != target_onnx and default_onnx.exists():
        import shutil
        target_onnx.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(default_onnx), str(target_onnx))
        logger.info(f"ONNX 模型已移动到: {target_onnx}")
    else:
        logger.info(f"ONNX 模型已保存: {default_onnx}")


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型 ONNX 导出")
    parser.add_argument("-m", "--model", required=True, help="输入 PyTorch 模型路径 (.pt)")
    parser.add_argument("-o", "--output", required=True, help="输出 ONNX 模型路径 (.onnx)")
    parser.add_argument("--imgsz", type=int, default=640, help="模型输入尺寸（默认 640）")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本（默认 12）")
    parser.add_argument("--no-simplify", action="store_true",
                        help="不执行 ONNX 图简化")
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()

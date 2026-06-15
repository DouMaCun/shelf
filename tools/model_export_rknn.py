# -*- coding: utf-8 -*-
"""
ONNX 模型 -> RKNN 转换脚本（在开发机 x86 上运行）。

工具链说明
----------
此脚本运行在**开发机**（x86/Ubuntu），不在 RK3588 设备上运行。
转换完成后将 .rknn 文件拷贝到设备，设备侧使用 RKNNDetector 加载推理。

开发机（x86/Ubuntu）：
    pip install rknn-toolkit2
    python tools/model_export_rknn.py -i models/yolo11n.onnx -o models/yolo11n.rknn

RK3588 设备（ARM64）：
    pip install rknn-toolkit-lite2
    # 修改 config/default.yaml:
    #   detector.backend: rknn
    #   detector.model_path: models/yolo11n.rknn
    python -m server.app

参数说明
--------
  -i, --input    : 输入 ONNX 模型路径
  -o, --output   : 输出 RKNN 模型路径
  --target       : 目标平台，默认 rk3588
  --int8         : 启用 INT8 量化（需配合 --calib-dir）
  --calib-dir    : INT8 校准图像目录（jpg/png，建议 50-200 张代表性图片）
  --imgsz        : 模型输入尺寸，默认 640
  --mean         : 输入归一化均值，默认 0,0,0
  --std          : 输入归一化标准差，默认 255,255,255（等价于 /255）

INT8 量化说明
-------------
量化可将推理速度提升约 2x，模型体积缩小约 4x，精度损失通常 < 1% mAP。
校准集建议使用与实际部署场景相似的货架图片，50 张即可，越多越准。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from loguru import logger


def convert_to_rknn(
    input_path: str,
    output_path: str,
    target: str = "rk3588",
    do_int8: bool = False,
    calib_dir: str = "",
    imgsz: int = 640,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> None:
    """将 ONNX 模型转换为 RKNN 格式。

    Args:
        input_path: 输入 ONNX 模型路径
        output_path: 输出 RKNN 模型路径
        target: 目标平台，如 "rk3588"、"rk3566"
        do_int8: 是否启用 INT8 量化
        calib_dir: INT8 校准图像目录（do_int8=True 时必须提供）
        imgsz: 模型输入尺寸（正方形）
        mean: 输入归一化均值，默认 [0, 0, 0]
        std: 输入归一化标准差，默认 [255, 255, 255]
    """
    try:
        from rknn.api import RKNN
    except ImportError:
        raise ImportError(
            "未找到 rknn-toolkit2，请在开发机（x86）上安装：\n"
            "  pip install rknn-toolkit2\n"
            "注意：rknn-toolkit2 仅支持 x86/Ubuntu，不能在 RK3588 设备上安装。"
        )

    if mean is None:
        mean = [0, 0, 0]
    if std is None:
        std = [255, 255, 255]

    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    rknn = RKNN(verbose=False)

    # ---- 1. 配置模型转换参数 ----
    logger.info(f"配置转换参数: target={target}, int8={do_int8}")
    ret = rknn.config(
        mean_values=[mean],
        std_values=[std],
        target_platform=target,
        quantized_algorithm="normal",
        quantized_method="channel",
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config 失败，错误码: {ret}")

    # ---- 2. 加载 ONNX 模型 ----
    logger.info(f"加载 ONNX 模型: {input_path}")
    ret = rknn.load_onnx(
        model=input_path,
        input_size_list=[[1, 3, imgsz, imgsz]],
    )
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx 失败，错误码: {ret}")

    # ---- 3. 构建 RKNN 模型 ----
    logger.info("构建 RKNN 模型...")
    ret = rknn.build(do_quantization=do_int8)
    if ret != 0:
        raise RuntimeError(f"rknn.build 失败，错误码: {ret}")

    # ---- 4. INT8 量化校准（可选）----
    if do_int8:
        if not calib_dir:
            raise ValueError("启用 INT8 量化时必须提供 --calib-dir")
        calib_list = _collect_calib_images(calib_dir)
        logger.info(f"INT8 量化校准: {len(calib_list)} 张图片")
        ret = rknn.hybrid_quantization_step1(dataset=calib_list)
        if ret != 0:
            raise RuntimeError(f"量化校准失败，错误码: {ret}")

    # ---- 5. 导出 RKNN 模型 ----
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"导出 RKNN 模型: {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn 失败，错误码: {ret}")

    rknn.release()
    logger.info(f"转换完成: {output_path}")
    logger.info(f"请将 {output_path} 拷贝到 RK3588 设备，并修改配置文件：")
    logger.info(f"  detector.backend: rknn")
    logger.info(f"  detector.model_path: {Path(output_path).name}")


def _collect_calib_images(calib_dir: str) -> str:
    """收集校准图像列表，写入临时文件供 RKNN 使用。"""
    import tempfile

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        str(p) for p in Path(calib_dir).iterdir()
        if p.suffix.lower() in exts
    ]
    if not images:
        raise ValueError(f"校准目录中未找到图片: {calib_dir}")

    # RKNN 需要一个包含图片路径的文本文件
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write("\n".join(images))
    tmp.close()
    logger.info(f"校准图像列表: {tmp.name} ({len(images)} 张)")
    return tmp.name


def main():
    parser = argparse.ArgumentParser(
        description="ONNX -> RKNN 模型转换工具（在开发机 x86 上运行）"
    )
    parser.add_argument("-i", "--input", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("-o", "--output", required=True, help="输出 RKNN 模型路径")
    parser.add_argument("--target", default="rk3588",
                        help="目标平台（默认 rk3588，其他如 rk3566、rk3568）")
    parser.add_argument("--int8", action="store_true", help="启用 INT8 量化")
    parser.add_argument("--calib-dir", default="",
                        help="INT8 校准图像目录（启用 --int8 时必须提供）")
    parser.add_argument("--imgsz", type=int, default=640, help="模型输入尺寸（默认 640）")
    parser.add_argument("--mean", default="0,0,0",
                        help="输入归一化均值，逗号分隔（默认 0,0,0）")
    parser.add_argument("--std", default="255,255,255",
                        help="输入归一化标准差，逗号分隔（默认 255,255,255）")
    args = parser.parse_args()

    mean = [float(x) for x in args.mean.split(",")]
    std = [float(x) for x in args.std.split(",")]

    convert_to_rknn(
        input_path=args.input,
        output_path=args.output,
        target=args.target,
        do_int8=args.int8,
        calib_dir=args.calib_dir,
        imgsz=args.imgsz,
        mean=mean,
        std=std,
    )


if __name__ == "__main__":
    main()

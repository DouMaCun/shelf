# -*- coding: utf-8 -*-
"""
数据集构建脚本。

功能：
  从采集的货架监控视频中抽取帧，用于 YOLO 模型训练。
  支持按固定帧间隔抽帧，避免冗余。

使用方式：
  python tools/dataset_builder.py -i data/raw/video.mp4 -o data/raw/frames -n 5

参数说明：
  -i, --input: 输入视频文件路径
  -o, --output: 输出帧目录
  -n, --every: 每隔 N 帧保存一帧（默认 5）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from loguru import logger


def extract_frames(video_path: str, output_dir: str, every_n: int = 5) -> None:
    """从视频中按固定帧间隔抽取帧。

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        every_n: 每隔 N 帧保存一帧
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频总帧数: {total_frames}, 每隔 {every_n} 帧抽取一帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            fname = f"frame_{frame_idx:06d}.jpg"
            fpath = out_path / fname
            cv2.imwrite(str(fpath), frame)
            saved_count += 1

            if saved_count % 100 == 0:
                logger.info(f"已抽取 {saved_count} 帧 (进度 {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    logger.info(f"抽取完成: 总共保存 {saved_count} 帧到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="数据集构建 - 视频帧抽取")
    parser.add_argument("-i", "--input", required=True, help="输入视频文件路径")
    parser.add_argument("-o", "--output", default="data/raw/frames", help="输出帧目录")
    parser.add_argument("-n", "--every", type=int, default=5,
                        help="每隔 N 帧保存一帧（默认 5）")
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.every)


if __name__ == "__main__":
    main()

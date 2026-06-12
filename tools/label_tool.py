# -*- coding: utf-8 -*-
"""
货架 RoI 标注工具。

功能：
  通过 OpenCV GUI 交互式标注货架槽位区域。
  标注结果直接写入 config/shelf_layout.yaml，供推理管线使用。

使用方式：
  python tools/label_tool.py -i data/raw/shelf_photo.jpg -s config/shelf_layout.yaml

操作说明：
  - 左键拖拽：绘制槽位矩形框
  - 按 's' 键：保存当前标注到配置文件
  - 按 'd' 键：删除最后一个标注框
  - 按 'q' 键：退出
  - 标注框上方会显示槽位 ID 标签
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np


def load_image(path: str) -> np.ndarray:
    """加载图像并调整大小以适应屏幕。

    如果图像宽度超过 1920 或高度超过 1080，按比例缩小。
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    h, w = img.shape[:2]
    max_w, max_h = 1920, 1080
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def draw_rects(img: np.ndarray, slots: list[dict], current_rect: tuple | None = None) -> np.ndarray:
    """在图像上绘制已标注的槽位和当前正在绘制的矩形。

    Args:
        img: 原始图像
        slots: 已标注的槽位列表 [{id, roi, sku_id}]
        current_rect: 当前正在拖拽的矩形 (x1, y1, x2, y2) 或 None

    Returns:
        带标注的可视化图像
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    # 绘制已确认的槽位
    for slot in slots:
        roi = slot['roi']
        x1 = int(roi[0] * w)
        y1 = int(roi[1] * h)
        x2 = int(roi[2] * w)
        y2 = int(roi[3] * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 显示槽位 ID
        cv2.putText(vis, f"{slot['id']} ({slot.get('sku_id', '')})",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 绘制正在绘制的矩形（虚线效果用半透明实线替代）
    if current_rect is not None:
        x1, y1, x2, y2 = current_rect
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return vis


def save_slots(slots: list[dict], output_path: str, img_w: int, img_h: int) -> None:
    """将槽位信息写入 YAML 配置文件。

    自动将像素坐标转为归一化坐标。

    Args:
        slots: 槽位列表
        output_path: 输出 YAML 文件路径
        img_w: 图像宽度
        img_h: 图像高度
    """
    normalized_slots = []
    for s in slots:
        roi = s['roi']  # 像素坐标
        normalized_slots.append({
            'id': s['id'],
            'roi': [
                round(roi[0] / img_w, 4),
                round(roi[1] / img_h, 4),
                round(roi[2] / img_w, 4),
                round(roi[3] / img_h, 4),
            ],
            'sku_id': s.get('sku_id', f"sku_{s['id']}"),
            'description': s.get('description', ''),
        })

    config = {
        'shelf_id': 'shelf_01',
        'camera_id': 'cam_01',
        'width': img_w,
        'height': img_h,
        'slots': normalized_slots,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"已保存 {len(normalized_slots)} 个槽位到 {output_path}")


def main():
    """标注工具主函数。

    交互式窗口循环：
    - 鼠标左键按下开始拖拽
    - 鼠标移动时更新矩形
    - 鼠标左键松开时确认矩形（保存像素坐标）
    """
    parser = argparse.ArgumentParser(description="货架槽位 RoI 标注工具")
    parser.add_argument("-i", "--input", required=True, help="货架照片路径")
    parser.add_argument("-s", "--shelf", default="config/shelf_layout.yaml",
                        help="货架布局配置输出路径")
    args = parser.parse_args()

    img = load_image(args.input)
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    print("操作: 拖拽左键画框 | 's' 保存 | 'd' 删除上一个 | 'q' 退出")

    # 槽位列表：{id, roi(像素), sku_id}
    slots: list[dict] = []
    next_id = 1

    # 交互状态
    drawing = False
    start_x, start_y = 0, 0
    current_rect = None  # (x1, y1, x2, y2)

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_x, start_y, current_rect, slots, next_id

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                current_rect = (start_x, start_y, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_rect = None
            x1, y1 = min(start_x, x), min(start_y, y)
            x2, y2 = max(start_x, x), max(start_y, y)
            if x2 - x1 < 10 or y2 - y1 < 10:
                return  # 忽略过小的框

            slots.append({
                'id': f"slot_{next_id:02d}",
                'roi': [x1, y1, x2, y2],  # 暂存像素坐标
                'sku_id': f"sku_{next_id:02d}",
            })
            print(f"添加槽位 slot_{next_id:02d}: ({x1}, {y1}) -> ({x2}, {y2})")
            next_id += 1

    cv2.namedWindow("Shelf Label Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Shelf Label Tool", mouse_callback)

    while True:
        vis = draw_rects(img, slots, current_rect)
        cv2.imshow("Shelf Label Tool", vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_slots(slots, args.shelf, img_w, img_h)
        elif key == ord('d'):
            if slots:
                removed = slots.pop()
                print(f"已删除槽位 {removed['id']}")
                next_id -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

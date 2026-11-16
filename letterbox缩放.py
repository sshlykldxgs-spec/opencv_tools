import cv2 as cv
import numpy as np

# YOLO 输入要求固定边长（如 640）。直接 resize 会拉伸变形；letterbox = 等比缩放 + 灰色补边，目标不变形。
# 训练和推理必须用同一种预处理，否则精度会掉。


def letterbox(img, new=640, color=(114, 114, 114), stride=32, auto_pad=False):
    """等比缩放到长边 = new，再补边。返回 (图, 缩放比例 r, 左补边 dx, 上补边 dy)。
    auto_pad=True：只补到 stride 的整数倍（推理时更快）；False：补成 new×new 的正方形（训练常用）"""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if auto_pad:
        pw, ph = (new - nw) % stride, (new - nh) % stride
    else:
        pw, ph = new - nw, new - nh
    dx, dy = pw / 2, ph / 2
    interp = cv.INTER_AREA if r < 1 else cv.INTER_LINEAR  # 缩小用 INTER_AREA 不会产生锯齿/摩尔纹
    if (nw, nh) != (w, h):
        img = cv.resize(img, (nw, nh), interpolation=interp)
    top, bottom = int(round(dy - 0.1)), int(round(dy + 0.1))
    left, right = int(round(dx - 0.1)), int(round(dx + 0.1))
    out = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return out, r, left, top


def boxes_to_letterbox(boxes_xyxy, r, dx, dy):
    """原图像素坐标 (x1,y1,x2,y2) → letterbox 图坐标"""
    b = np.asarray(boxes_xyxy, np.float64).reshape(-1, 4) * r
    b[:, [0, 2]] += dx
    b[:, [1, 3]] += dy
    return b


def boxes_from_letterbox(boxes_xyxy, r, dx, dy, orig_shape):
    """模型输出（letterbox 图坐标）→ 还原到原图坐标，并裁剪到图内"""
    b = np.asarray(boxes_xyxy, np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - dx) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dy) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, orig_shape[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, orig_shape[0])
    return b


if __name__ == '__main__':
    img = np.full((480, 800, 3), 200, np.uint8)
    cv.rectangle(img, (100, 100), (300, 250), (0, 0, 255), -1)
    out, r, dx, dy = letterbox(img, 640)
    box = boxes_to_letterbox([[100, 100, 300, 250]], r, dx, dy)
    print('输出尺寸', out.shape, '缩放比例', round(r, 3), '补边', (dx, dy))
    print('框坐标 原图 → letterbox 图：', box.round(1).tolist())
    print('还原回原图：', boxes_from_letterbox(box, r, dx, dy, img.shape).round(1).tolist())

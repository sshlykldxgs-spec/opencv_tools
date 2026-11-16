import os
import glob
import random
import cv2 as cv
import numpy as np

# ===== 配置（批量生成增强数据时用）=====
IMG_DIR = 'dataset/images'       # 原图
LBL_DIR = 'dataset/labels'       # YOLO 格式标注：每行 "类别 xc yc w h"，坐标已按图宽高归一化到 0~1
OUT_DIR = 'dataset_aug'          # 输出（自动建 images/ labels/）
COPIES = 3                       # 每张原图生成几张增强图
SIZE = 640                       # 输出边长
P_MOSAIC = 0.5                   # 用马赛克（4 张拼 1 张）的概率
FLIP_UD = 0.0                    # 上下翻转概率：俯拍/无人机画面可设 0.5，普通地面拍摄保持 0


# ---------- 标注读写 / 坐标转换 ----------
def read_labels(path):
    """读 YOLO txt → N×5 数组 [cls, xc, yc, w, h]；文件不存在或为空 → 0×5"""
    if not os.path.exists(path):
        return np.zeros((0, 5), np.float64)
    a = np.loadtxt(path, ndmin=2)
    return a.reshape(-1, 5) if a.size else np.zeros((0, 5), np.float64)


def write_labels(path, lab):
    with open(path, 'w') as f:
        for c, x, y, w, h in lab:
            f.write(f'{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')


def yolo_to_xyxy(lab, w, h):
    """归一化 (xc,yc,w,h) → 像素 (x1,y1,x2,y2)"""
    xc, yc, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    return np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)


def xyxy_to_yolo(cls, xyxy, w, h):
    x1, y1, x2, y2 = xyxy.T
    return np.stack([cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], 1)


# ---------- 单项增强 ----------
def hsv_jitter(img, h=0.015, s=0.7, v=0.4):
    """颜色抖动：色调/饱和度/亮度随机变化（对应不同光照、相机白平衡）。不影响标注框"""
    r = np.random.uniform(-1, 1, 3) * (h, s, v) + 1
    hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    return cv.cvtColor(cv.merge((cv.LUT(hue, lut_h), cv.LUT(sat, lut_s), cv.LUT(val, lut_v))), cv.COLOR_HSV2BGR)


def flip(img, lab, lr=0.5, ud=0.0):
    if random.random() < lr:
        img = img[:, ::-1]
        lab = lab.copy(); lab[:, 1] = 1 - lab[:, 1]
    if random.random() < ud:
        img = img[::-1]
        lab = lab.copy(); lab[:, 2] = 1 - lab[:, 2]
    return np.ascontiguousarray(img), lab


def random_affine(img, lab, out_size=640, degrees=10, translate=0.1, scale=(0.5, 1.5), shear=2.0,
                  perspective=0.0, min_area_ratio=0.2, fill=114):
    """随机旋转 + 缩放 + 平移 + 错切（+ 可选透视），输出 out_size×out_size。
    标注框：把原框 4 个角点同样变换，取外接矩形，再裁剪到图内；
    裁剪后面积不足原来 min_area_ratio、或宽高 < 2 像素的框丢掉（目标被切掉大半，留着只会教错模型）。
    注意：目标旋转后，外接矩形会比目标本身松（旋转 10° 内影响很小；想要更紧的框需要分割标注）。"""
    h, w = img.shape[:2]
    C = np.eye(3); C[0, 2], C[1, 2] = -w / 2, -h / 2
    P = np.eye(3); P[2, 0], P[2, 1] = random.uniform(-perspective, perspective), random.uniform(-perspective, perspective)
    a, sc = random.uniform(-degrees, degrees), random.uniform(*scale)
    R = np.eye(3); R[:2] = cv.getRotationMatrix2D((0, 0), a, sc)
    S = np.eye(3)
    S[0, 1] = np.tan(np.deg2rad(random.uniform(-shear, shear)))
    S[1, 0] = np.tan(np.deg2rad(random.uniform(-shear, shear)))
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    M = T @ S @ R @ P @ C
    col = (fill,) * 3
    if perspective:
        out = cv.warpPerspective(img, M, (out_size, out_size), flags=cv.INTER_LINEAR, borderValue=col)
    else:
        out = cv.warpAffine(img, M[:2], (out_size, out_size), flags=cv.INTER_LINEAR, borderValue=col)
    if len(lab) == 0:
        return out, lab
    b = yolo_to_xyxy(lab, w, h)
    pts = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1).reshape(-1, 2)
    pts = np.c_[pts, np.ones(len(pts))] @ M.T
    pts = (pts[:, :2] / pts[:, 2:3]).reshape(-1, 4, 2)
    nb = np.c_[pts[..., 0].min(1), pts[..., 1].min(1), pts[..., 0].max(1), pts[..., 1].max(1)]
    area0 = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
    nb[:, [0, 2]] = nb[:, [0, 2]].clip(0, out_size)
    nb[:, [1, 3]] = nb[:, [1, 3]].clip(0, out_size)
    bw, bh = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
    ok = (bw > 2) & (bh > 2) & (bw * bh > min_area_ratio * np.maximum(area0, 1e-9))
    return out, xyxy_to_yolo(lab[ok, 0], nb[ok], out_size, out_size)


def mosaic4(samples, size=640):
    """马赛克：4 张图拼成一张（一张图里同时出现多种背景、多个目标、目标尺度更小，对小目标检测很有帮助）。
    samples：4 个 (img, lab)。返回 2size×2size 的大图和标注，后面再接 random_affine 裁成 size×size"""
    canvas = np.full((2 * size, 2 * size, 3), 114, np.uint8)
    xc, yc = [int(random.uniform(0.5 * size, 1.5 * size)) for _ in range(2)]
    labs = []
    for i, (img, lab) in enumerate(samples):
        h, w = img.shape[:2]
        r = size / max(h, w)
        img = cv.resize(img, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv.INTER_AREA if r < 1 else cv.INTER_LINEAR)
        h, w = img.shape[:2]
        # 第 i 张图贴在拼接点 (xc,yc) 的左上/右上/左下/右下，靠拼接点的那个角对齐
        x1 = xc - w if i in (0, 2) else xc
        y1 = yc - h if i in (0, 1) else yc
        x1c, y1c, x2c, y2c = max(x1, 0), max(y1, 0), min(x1 + w, 2 * size), min(y1 + h, 2 * size)
        canvas[y1c:y2c, x1c:x2c] = img[y1c - y1:y2c - y1, x1c - x1:x2c - x1]
        if len(lab):
            b = yolo_to_xyxy(lab, w, h) + [x1, y1, x1, y1]
            b[:, [0, 2]] = b[:, [0, 2]].clip(x1c, x2c)  # 被画布/邻图边界切掉的部分按实际保留的范围裁剪
            b[:, [1, 3]] = b[:, [1, 3]].clip(y1c, y2c)
            full = yolo_to_xyxy(lab, w, h)
            keep = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) > 0.2 * (full[:, 2] - full[:, 0]) * (full[:, 3] - full[:, 1])
            keep &= ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)
            labs.append(xyxy_to_yolo(lab[keep, 0], b[keep], 2 * size, 2 * size))
    lab = np.concatenate(labs) if labs else np.zeros((0, 5))
    return canvas, lab


def augment(sample, pool, size=640, p_mosaic=0.5, flip_ud=0.0):
    """完整流程：[马赛克] → 随机仿射 → 翻转 → 颜色抖动。sample=(img, lab)，pool=可供拼马赛克的样本列表"""
    img, lab = sample
    if random.random() < p_mosaic and len(pool) >= 3:
        others = random.sample(pool, 3)
        big, blab = mosaic4([sample] + others, size)
        img, lab = random_affine(big, blab, size, scale=(0.5, 1.5), translate=0.1)
    else:
        img, lab = random_affine(img, lab, size, scale=(0.6, 1.4))
    img, lab = flip(img, lab, 0.5, flip_ud)
    return hsv_jitter(img), lab


def draw_boxes(img, lab):
    out = img.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2), c in zip(yolo_to_xyxy(lab, w, h), lab[:, 0]):
        cv.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.putText(out, str(int(c)), (int(x1), max(12, int(y1) - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


if __name__ == '__main__':
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    paths = sorted(p for e in exts for p in glob.glob(os.path.join(IMG_DIR, e)))
    if not paths:
        raise SystemExit('没找到图片：' + IMG_DIR)
    pool = []
    for p in paths:
        img = cv.imread(p)
        if img is None:
            continue
        pool.append((img, read_labels(os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + '.txt'))))
    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels'), exist_ok=True)
    n = 0
    for p, sample in zip(paths, pool):
        stem = os.path.splitext(os.path.basename(p))[0]
        for k in range(COPIES):
            img, lab = augment(sample, pool, SIZE, P_MOSAIC, FLIP_UD)
            cv.imwrite(os.path.join(OUT_DIR, 'images', f'{stem}_aug{k}.jpg'), img, [cv.IMWRITE_JPEG_QUALITY, 95])
            write_labels(os.path.join(OUT_DIR, 'labels', f'{stem}_aug{k}.txt'), lab)
            n += 1
    print(f'生成 {n} 张增强图 → {OUT_DIR}/（先用"标注检查与可视化.py"抽查几张再拿去训练）')

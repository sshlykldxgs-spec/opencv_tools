import os
import glob
import shutil
import cv2 as cv
import numpy as np

# 把前面学的 OpenCV 特征（对比度增强、边缘/梯度、纹理）合成一张 3 通道"特征图"，当作 YOLO 的输入图像。
# 特征图只做逐像素处理、不改变几何，所以原来的标注 txt 可以原封不动复制使用。
#
# 什么时候有用：目标和背景颜色接近（猫与草地/树影、迷彩）、光照差、纹理/轮廓比颜色更可靠的场景。
# 注意：YOLO 的预训练权重是在普通 RGB 照片上训练的，输入换成特征图后，预训练权重的帮助会变小。
# 是否真的涨点，要用同一份数据分别训练"原图"和"特征图"两个模型，比较验证集 mAP——不要凭感觉。

IMG_DIR = 'dataset/images'
LBL_DIR = 'dataset/labels'
OUT_DIR = 'dataset_feat'
MODE = 'gray_edge_lbp'     # 见 FEATURE_MODES


def lbp_uniform(gray, radius=1, n_points=8, tol=4.0, blur_sigma=0.7):
    """旋转不变均匀 LBP（只用 OpenCV+numpy），同"lbp-纹路-人脸检测2.0.py"。输出标签 0~n_points+1"""
    g = gray.astype(np.float32)
    if blur_sigma > 0:
        g = cv.GaussianBlur(g, (0, 0), blur_sigma)
    h, w = g.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    bits = np.empty((n_points, h, w), np.uint8)
    for p in range(n_points):
        a = 2 * np.pi * p / n_points
        nb = cv.remap(g, (xs + radius * np.cos(a)).astype(np.float32), (ys - radius * np.sin(a)).astype(np.float32),
                      cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        bits[p] = (nb - g >= -tol)
    trans = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    return np.where(trans <= 2, bits.sum(axis=0), n_points + 1).astype(np.uint8)


def clahe_gray(gray, clip=2.0, grid=8):
    """自适应直方图均衡（限制对比度）：暗处/逆光处的细节拉出来，又不像全局均衡那样把噪声放大"""
    return cv.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def gradient_mag(gray):
    """Scharr 梯度幅值，归一化到 0~255。先轻度高斯平滑减少噪声边缘。
    用图像自身的 99.5% 分位数做上限归一化（而不是固定 255），不同亮度的图得到的边缘强度范围一致"""
    g = cv.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    mag = cv.magnitude(cv.Scharr(g, cv.CV_32F, 1, 0), cv.Scharr(g, cv.CV_32F, 0, 1))
    top = max(float(np.percentile(mag, 99.5)), 1e-6)
    return np.clip(mag / top * 255, 0, 255).astype(np.uint8)


def lbp_channel(gray):
    return (lbp_uniform(gray).astype(np.float32) / 9 * 255).astype(np.uint8)  # 标签 0~9 拉伸到 0~255


FEATURE_MODES = {
    # 模式名: (通道0, 通道1, 通道2)，按 BGR 顺序存；训练时 YOLO 当普通 3 通道图读
    'gray_edge_lbp': lambda bgr, gray: (clahe_gray(gray), gradient_mag(gray), lbp_channel(gray)),  # 增强灰度 + 边缘 + 纹理
    'clahe_rgb':     lambda bgr, gray: tuple(cv.split(_clahe_color(bgr))),                          # 只做对比度增强，保留颜色
    'gray_edge':     lambda bgr, gray: (clahe_gray(gray), gradient_mag(gray), gray),                # 增强灰度 + 边缘 + 原灰度
}


def _clahe_color(bgr):
    """只对亮度做 CLAHE，颜色基本不变（直接对 B/G/R 分别均衡会偏色）"""
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe_gray(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def make_feature_image(bgr, mode=MODE):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    return cv.merge(FEATURE_MODES[mode](bgr, gray))


if __name__ == '__main__':
    paths = sorted(p for e in ('*.jpg', '*.jpeg', '*.png', '*.bmp') for p in glob.glob(os.path.join(IMG_DIR, e)))
    if not paths:
        raise SystemExit('没找到图片：' + IMG_DIR)
    os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels'), exist_ok=True)
    for p in paths:
        img = cv.imread(p)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        cv.imwrite(os.path.join(OUT_DIR, 'images', stem + '.png'), make_feature_image(img))  # 存 png，避免 jpg 压缩破坏纹理/边缘
        lp = os.path.join(LBL_DIR, stem + '.txt')
        if os.path.exists(lp):
            shutil.copy2(lp, os.path.join(OUT_DIR, 'labels', stem + '.txt'))
    print(f'已生成 {len(paths)} 张特征图（模式 {MODE}）→ {OUT_DIR}/，标注原样复制')

import os
import cv2 as cv
import numpy as np

# ===== 配置 =====
VIDEO_PATH = 'video.mp4'      # 输入视频
OUT_DIR = 'frames'            # 输出文件夹
EVERY = 5                     # 每隔多少帧取一帧
BLUR_RATIO = 0.5              # 清晰度低于"中位数 × 该比例"的帧丢掉（手机晃动、对焦失败的帧）
DUP_BITS = 6                  # 和上一张保留的帧相比，哈希差异位数 ≤ 该值就算重复（0~64，越大去重越狠）


def sharpness(gray):
    """清晰度：拉普拉斯方差。先统一缩到宽 640，这样不同分辨率的视频数值可比"""
    s = 640 / gray.shape[1]
    if s < 1:
        gray = cv.resize(gray, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def dhash(gray, size=8):
    """差值哈希：缩成 9x8，比较相邻像素的左右大小，得到 64 位指纹。轻微亮度变化、压缩噪声不影响指纹"""
    small = cv.resize(gray, (size + 1, size), interpolation=cv.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()


def extract(video, out_dir, every=5, blur_ratio=0.5, dup_bits=6):
    """返回 (保存的帧数, 因模糊丢弃数, 因重复丢弃数)。
    做两遍：第一遍只算每个候选帧的清晰度和哈希（内存里只存数字），第二遍把保留下来的帧写盘。
    清晰度阈值用整段视频的中位数来定，所以不用手动调，也不会因为视频整体偏暗/偏糊而全部丢光。"""
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit('视频打不开：' + video)
    idx, info = 0, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % every == 0:
            g = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
            info.append((idx, sharpness(g), dhash(g)))
        idx += 1
    cap.release()
    if not info:
        return 0, 0, 0
    thr = blur_ratio * np.median([i[1] for i in info])
    keep, blur, dup, last = [], 0, 0, None
    for i, sh, h in info:
        if sh < thr:
            blur += 1
            continue
        if last is not None and np.count_nonzero(h != last) <= dup_bits:
            dup += 1
            continue
        keep.append(i)
        last = h
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]  # 文件名带上视频名：后面划分数据集时可以按视频分组
    keep_set, idx = set(keep), 0
    cap = cv.VideoCapture(video)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx in keep_set:
            cv.imwrite(os.path.join(out_dir, f'{stem}_{idx:06d}.jpg'), f, [cv.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    return len(keep), blur, dup


if __name__ == '__main__':
    n, blur, dup = extract(VIDEO_PATH, OUT_DIR, EVERY, BLUR_RATIO, DUP_BITS)
    print(f'保存 {n} 张；因模糊丢弃 {blur} 张；因重复丢弃 {dup} 张  →  {OUT_DIR}/')

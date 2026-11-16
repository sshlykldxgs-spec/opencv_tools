import os
import glob
import collections
import cv2 as cv
import numpy as np

# ===== 配置 =====
IMG_DIR = 'dataset/images'
LBL_DIR = 'dataset/labels'
NUM_CLASSES = None        # 类别数（填了就会检查类别编号是否越界）；None 则不检查
SAVE_VIS_DIR = 'check_vis'  # 有问题的图 + 随机抽样图会画好框存到这里；None 则不保存
SAMPLE = 20               # 随机抽几张画框看看
MIN_BOX_PX = 4            # 框宽或高小于这个像素数，报"过小"


def check_dataset(img_dir, lbl_dir, num_classes=None, min_box_px=4):
    """检查 YOLO 数据集，返回 (问题列表, 每类框数, 统计)。训练前跑一遍，能省掉"训练很久才发现标注有问题"：
      · 图没有标注文件 / 标注没有对应的图
      · 一行不是 5 个数、类别不是整数、类别越界
      · 坐标不在 0~1（常见错误：标注工具导出了像素坐标）
      · 框宽高 ≤ 0 或小于 min_box_px 像素
      · 同一张图里完全重复的框"""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(exts)}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}
    problems, per_class, widths, heights, empty = [], collections.Counter(), [], [], 0
    for stem in sorted(set(lbls) - set(imgs)):
        problems.append((stem + '.txt', '有标注文件但找不到对应的图'))
    for stem, fname in sorted(imgs.items()):
        lp = os.path.join(lbl_dir, stem + '.txt')
        if not os.path.exists(lp):
            problems.append((fname, '没有标注文件（会被当成纯背景图；如果是故意的可以忽略）'))
            continue
        img = cv.imread(os.path.join(img_dir, fname))
        if img is None:
            problems.append((fname, '图片读取失败（损坏？）'))
            continue
        ih, iw = img.shape[:2]
        rows = [r.split() for r in open(lp, encoding='utf-8').read().strip().splitlines() if r.strip()]
        if not rows:
            empty += 1
        seen = set()
        for i, r in enumerate(rows, 1):
            tag = f'{stem}.txt 第{i}行'
            if len(r) != 5:
                problems.append((tag, f'应有 5 个数，实际 {len(r)} 个（分割标注/带置信度的格式？）'))
                continue
            try:
                c, x, y, w, h = float(r[0]), *map(float, r[1:])
            except ValueError:
                problems.append((tag, '含有非数字内容'))
                continue
            if c != int(c) or c < 0 or (num_classes is not None and c >= num_classes):
                problems.append((tag, f'类别编号不合法：{r[0]}'))
                continue
            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                problems.append((tag, f'坐标不在 0~1：{r[1:]}（导出了像素坐标？）'))
                continue
            if w <= 0 or h <= 0:
                problems.append((tag, '宽或高 ≤ 0'))
                continue
            if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                problems.append((tag, '框超出图像边界'))
            if w * iw < min_box_px or h * ih < min_box_px:
                problems.append((tag, f'框过小：{w * iw:.1f}×{h * ih:.1f} 像素'))
            key = tuple(r)
            if key in seen:
                problems.append((tag, '与前面某行完全重复'))
            seen.add(key)
            per_class[int(c)] += 1
            widths.append(w * iw); heights.append(h * ih)
    stats = dict(images=len(imgs), empty=empty, boxes=sum(per_class.values()),
                 w_med=float(np.median(widths)) if widths else 0, h_med=float(np.median(heights)) if heights else 0)
    return problems, per_class, stats


def draw_one(img_path, lbl_path):
    img = cv.imread(img_path)
    h, w = img.shape[:2]
    if os.path.exists(lbl_path):
        for r in open(lbl_path).read().strip().splitlines():
            p = r.split()
            if len(p) != 5:
                continue
            c, x, y, bw, bh = int(float(p[0])), *map(float, p[1:])
            x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img, str(c), (x1, max(12, y1 - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


if __name__ == '__main__':
    problems, per_class, st = check_dataset(IMG_DIR, LBL_DIR, NUM_CLASSES, MIN_BOX_PX)
    print(f"图片 {st['images']} 张（其中无目标的背景图 {st['empty']} 张），标注框 {st['boxes']} 个")
    print('每类框数：', dict(sorted(per_class.items())))
    if per_class:
        mx, mn = max(per_class.values()), min(per_class.values())
        if mx > 10 * mn:
            print(f'⚠ 类别严重不均衡（最多 {mx}，最少 {mn}），少数类建议补采或做针对性增强')
    print(f"框尺寸中位数：{st['w_med']:.0f}×{st['h_med']:.0f} 像素（太小的目标在 640 输入下可能检测不出，可考虑提高输入尺寸）")
    print(f'发现问题 {len(problems)} 个')
    for where, msg in problems[:50]:
        print('  ·', where, '→', msg)
    if SAVE_VIS_DIR:
        import random
        os.makedirs(SAVE_VIS_DIR, exist_ok=True)
        bad = {w.split('.txt')[0].split(' ')[0] for w, _ in problems}
        names = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        pick = [f for f in names if os.path.splitext(f)[0] in bad][:30] + random.sample(names, min(SAMPLE, len(names)))
        for f in dict.fromkeys(pick):
            cv.imwrite(os.path.join(SAVE_VIS_DIR, f), draw_one(os.path.join(IMG_DIR, f), os.path.join(LBL_DIR, os.path.splitext(f)[0] + '.txt')))
        print(f'已保存 {len(set(pick))} 张画好框的图到 {SAVE_VIS_DIR}/，打开看一眼框准不准')

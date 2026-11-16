import os
import re
import shutil
import random
import collections

# ===== 配置 =====
IMG_DIR = 'dataset/images'
LBL_DIR = 'dataset/labels'
OUT_DIR = 'yolo_data'
RATIOS = (0.8, 0.1, 0.1)       # 训练 / 验证 / 测试；测试集不要可以写 (0.9, 0.1, 0)
SEED = 0
GROUP_BY_VIDEO = True          # True：同一个视频抽出来的帧整体划进同一个集合（强烈建议，见下面说明）
CLASS_NAMES = ['cat']          # 写进 data.yaml


def group_key(stem):
    """文件名形如 视频名_000120 → 分组键是 视频名。抽帧脚本生成的文件名就是这个格式。"""
    stem = re.sub(r'_aug\d+$', '', stem)  # 增强出来的图和它的原图算同一组，避免"原图在训练集、增强图在验证集"
    m = re.match(r'^(.*)_\d+$', stem)
    return m.group(1) if m else stem


def split(stems, ratios=(0.8, 0.1, 0.1), seed=0, by_group=True):
    """返回 {'train': [...], 'val': [...], 'test': [...]}。
    为什么要按视频分组：同一段视频相邻帧几乎一样，如果随机打散，训练集里有一张、验证集里有它的"孪生帧"，
    验证指标会虚高（模型是背下来的，不是学会的），上线后才发现效果差很多。
    按组分配时用"最大的组先放、放进当前缺口最大的集合"的贪心，让各集合的图片数尽量接近比例。"""
    rnd = random.Random(seed)
    groups = collections.defaultdict(list)
    for s in stems:
        groups[group_key(s) if by_group else s].append(s)
    keys = list(groups)
    rnd.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    names = ['train', 'val', 'test']
    total = len(stems)
    target = {n: r * total for n, r in zip(names, ratios)}
    out = {n: [] for n in names}
    for k in keys:
        n = max((n for n in names if target[n] > 0), key=lambda n: target[n] - len(out[n]))
        out[n].extend(groups[k])
    return out


if __name__ == '__main__':
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = {os.path.splitext(f)[0]: f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)}
    parts = split(sorted(files), RATIOS, SEED, GROUP_BY_VIDEO)
    n_groups = len({group_key(s) for s in files})
    if GROUP_BY_VIDEO and n_groups < 10:
        print(f'⚠ 按文件名只分出 {n_groups} 组，划分可能很不均匀。图片不是视频抽帧、文件名只是编号（如 IMG_0001）时，请把 GROUP_BY_VIDEO 改成 False')
    print('提示：先划分、再只对 train 做数据增强；如果先增强再划分，增强图会泄漏到验证集。')
    for name, stems in parts.items():
        if not stems:
            continue
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(OUT_DIR, sub, name), exist_ok=True)
        for s in stems:
            shutil.copy2(os.path.join(IMG_DIR, files[s]), os.path.join(OUT_DIR, 'images', name, files[s]))
            lp = os.path.join(LBL_DIR, s + '.txt')
            dst = os.path.join(OUT_DIR, 'labels', name, s + '.txt')
            if os.path.exists(lp):
                shutil.copy2(lp, dst)
            else:
                open(dst, 'w').close()  # 没有标注文件 = 纯背景图，生成空 txt
        print(f'{name}: {len(stems)} 张')
    with open(os.path.join(OUT_DIR, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f'path: {os.path.abspath(OUT_DIR)}\ntrain: images/train\nval: images/val\n')
        if parts['test']:
            f.write('test: images/test\n')
        f.write(f'nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n')
    print('已生成', os.path.join(OUT_DIR, 'data.yaml'), '→ 训练：yolo detect train data=' + os.path.join(OUT_DIR, 'data.yaml') + ' model=yolo11n.pt')

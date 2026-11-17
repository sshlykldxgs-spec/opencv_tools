"""
水道漂浮物检测：数据预处理（检测框或分割，默认检测）
场景特点：目标小（瓶子、泡沫、落叶、塑料袋）、水面反光强、图大（无人机/岸边摄像头）→ 重点是切片(tiles)、反光增强(aug_glare)。
注意：deglare（去反光）变体会把"又白又亮"的区域抹掉，白色漂浮物可能被一起抹掉，只作对照实验用。

用法（Mac 上做预处理，只需要 opencv-python + numpy；装了 Pillow 会多一项 EXIF 检查）：
    1) 把下面 CFG 里的 img_dir / ann_path / label_format 改成你的，或者命令行覆盖：
       python 脚本.py --img 图片文件夹 --ann 标注(文件夹或 coco.json) --format yolo --out 输出 --workers 4
    2) 先跑一次 --check-only：只读取、严格检查、分组、划分并写报告（reports_check/），不生成数据集；看完报告再正式跑
    3) 正式跑：输出文件夹整个拷到 Windows，每个变体下的 data.yaml 直接用于 yolo 训练（路径是相对的）
标注格式：yolo / yolo_seg / coco / voc / mask(像素值=类别) / instmap(像素值=实例编号，SAM 系列常见)
这一版相对 2.0 的改动：类别从数据里自动得到（COCO 的 id 不连续也没问题）；每条标注严格检查并出报告；
按类别均衡划分（以"组"为单位）；视频分组不再只靠文件名；稀有类别自动多生成增强图；输出自检含泄漏检查。
"""

# ============ 配置：只改这里（或用命令行参数覆盖）============
CFG = dict(
    task='detect',                  # 'detect' 或 'segment'
    img_dir='dataset/images',
    ann_path='dataset/labels',      # txt/xml/png 文件夹，或 COCO 的 json 文件
    label_format='yolo',            # yolo / yolo_seg / coco / voc / mask / instmap
    classes_file=None,              # yolo 的类别名文件（classes.txt / obj.names / data.yaml）；COCO/VOC 自动取
    class_map={},                   # 合并同类，如 {'plastic_bottle': '塑料瓶', 'bottle': '塑料瓶'}
    drop_classes=[],
    out_dir='prep_water',
    size=640, tile_size=640, tile_overlap=0.2,
    copies=2, oversample=True,
    degrees=5, flip_lr=0.5, flip_ud=0.0, scale_lo=0.6, scale_hi=1.4,
    min_box_px=3,                   # 漂浮物小：3 像素以下才丢
    full_pix=['color', 'glare'],
    variants=['base', 'clahe', 'tiles', 'aug_geo', 'aug_color', 'aug_glare', 'aug_degrade', 'aug_full'],   # 可加 'deglare' 'hist_eq' 'edge_scharr' 'feat_stack' 'aug_mosaic' 'cls_crops'
)

# ===================== 0 公共基础：异常体系 / 日志 / 安全 IO / 容错进程池（标注前、标注后脚本共用）=====================
import os
import re
import sys
import csv
import json
import glob
import time
import zlib
import errno
import types
import shutil
import random
import hashlib
import argparse
import tempfile
import difflib
import itertools
import traceback
import collections
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np

try:                                   # Pillow 只用来读图片头里的尺寸/EXIF（没有也能跑，只是慢一点、少几项检查）
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None      # 只读文件头不解码，大航拍图不该被 Pillow 的"解压炸弹"警告拦住
except Exception:                      # noqa
    Image = None

__version__ = '3.1'


# ---------- 异常体系：所有"预期内的失败"都是 PrepError，带中文原因和处理建议；其余异常是 bug，会写 crash 日志 ----------
class PrepError(Exception):
    """可预期的失败（配置错、输入不对、输出写不了…）。exit_code=2"""
    exit_code = 2

    def __init__(self, msg, hint=''):
        super().__init__(msg)
        self.hint = hint


class ConfigError(PrepError):
    pass


class InputError(PrepError):
    pass


class OutputError(PrepError):
    pass


class Failure:
    """某一项处理失败的记录（不会让整批任务崩溃，最后汇总成报告）"""
    def __init__(self, msg, tb='', fatal=False, crashed=False):
        self.msg, self.tb, self.fatal, self.crashed = msg, tb, fatal, crashed

    def __repr__(self):
        return f'Failure({self.msg})'


FATAL_ERRNO = {errno.ENOSPC, errno.EROFS, getattr(errno, 'EDQUOT', -1)}     # 磁盘满/只读/配额：继续跑没有意义，直接终止


def os_error_hint(e):
    n = getattr(e, 'errno', None)
    if n == errno.ENOSPC:
        return '磁盘空间不足：清理空间，或把 out_dir 换到空间更大的盘'
    if n in (errno.EACCES, errno.EPERM):
        return '没有权限，或文件正被别的程序占用（Windows 上常见：Excel/预览窗口打开了 csv）。关掉占用它的程序，或换个输出文件夹'
    if n == errno.ENAMETOOLONG:
        return '路径太长（Windows 默认限制 260 个字符）：把数据放到更浅的文件夹，如 D:\\data'
    if n == errno.EROFS:
        return '目标位置是只读的'
    if n == errno.EMFILE:
        return '同时打开的文件太多：减小 workers'
    if n == errno.ENOENT:
        return '文件或文件夹不存在（路径写错，或文件被移走/删除了）'
    return ''


# ---------- 日志 ----------
LOGF = None
LOGBUF = collections.deque(maxlen=60)       # 最近的日志行：崩溃日志里带上，方便定位


def open_log(path):
    global LOGF
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    LOGF = open(path, 'a', encoding='utf-8')


def close_log():
    global LOGF
    if LOGF:
        try:
            LOGF.close()
        finally:
            LOGF = None


def log(msg=''):
    print(msg, flush=True)
    LOGBUF.append(str(msg))
    if LOGF:
        try:
            LOGF.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
            LOGF.flush()
        except Exception:                          # noqa  日志写不了不能影响主流程
            pass


# ---------- 安全 IO ----------
def retry_io(fn, tries=3, delay=0.3):
    """网络盘 / 云盘同步 / 杀毒软件扫描时，文件偶尔会瞬时打不开：重试几次。文件不存在不重试"""
    for i in range(tries):
        try:
            return fn()
        except OSError as e:
            if i == tries - 1 or e.errno in (errno.ENOENT, errno.ENOSPC, errno.EROFS):
                raise
            time.sleep(delay * (i + 1))


def read_bytes(path):
    return retry_io(lambda: np.fromfile(path, np.uint8))


def imread_u(path, flags=cv.IMREAD_COLOR):
    """cv.imread 在 Windows 上打不开含中文的路径：用 numpy 读字节再解码；读不了/解码失败返回 None"""
    try:
        data = read_bytes(path)
    except OSError:
        return None
    return cv.imdecode(data, flags) if data.size else None


def imwrite_u(path, img, params=None):
    ok, buf = cv.imencode(os.path.splitext(path)[1], img, params or [])
    if not ok:
        raise OSError(f'编码失败：{path}')
    retry_io(lambda: buf.tofile(path))


def read_text(path):
    """读文本：UTF-8（含 BOM）优先，失败按 GB18030 读（国内工具/Windows 记事本存的 ANSI 文件）。返回 (文本, 编码)"""
    raw = retry_io(lambda: open(path, 'rb').read())
    for enc in ('utf-8-sig', 'gb18030'):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise InputError(f'无法解码文本文件：{path}', '请另存为 UTF-8 编码')


def write_text(path, text):
    """原子写：先写临时文件再替换，中途崩溃不会留下写了一半的文件"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    os.replace(tmp, path)


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:       # utf-8-sig：Excel 直接打开中文不乱码
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def safe_name(s):
    return re.sub(r'[<>:"|?*\\\s]+', '_', s)


def key_to_stem(key):
    return key.replace('/', '+')


# ---------- 容错并行 ----------
class Guarded:
    """把任何函数包起来：单项出错（含内存不足）只返回 Failure，不会让整批任务崩溃。可被 pickle，能在子进程里用"""
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, item):
        try:
            return self.fn(item)
        except MemoryError:
            return Failure('内存不足（MemoryError）', '内存不足时可减小 workers、proc_max_side，或在 variants 里去掉 tiles/aug_mosaic')
        except OSError as e:
            return Failure(f'{type(e).__name__}: {e}', os_error_hint(e), fatal=e.errno in FATAL_ERRNO)
        except Exception as e:                                       # noqa
            return Failure(f'{type(e).__name__}: {e}', traceback.format_exc(limit=6))


def _isolate(g, items, idxs, results):
    """进程池整体崩溃后（某个文件触发了底层崩溃/被系统杀掉）：用单进程逐个重跑受影响的项，定位出真正的"元凶"，其余正常完成"""
    ex = None
    for i in idxs:
        if ex is None:
            ex = ProcessPoolExecutor(1, mp_context=mp.get_context('spawn'))
        try:
            results[i] = ex.submit(g, items[i]).result()
        except BrokenProcessPool:
            results[i] = Failure('处理进程崩溃（常见原因：内存不足，或这个文件触发了 OpenCV 的底层崩溃）', crashed=True)
            ex.shutdown(wait=False, cancel_futures=True)
            ex = None
        except Exception as e:                                       # noqa
            results[i] = Failure(f'{type(e).__name__}: {e}')
    if ex is not None:
        ex.shutdown()


def run_pool(fn, items, workers, desc=''):
    """按顺序返回每一项的结果；失败的项是 Failure 对象。workers>1 用 spawn 进程池（Windows/macOS/Linux 行为一致）"""
    g, n = Guarded(fn), len(items)
    results = [None] * n
    step = max(1, n // 10)
    if workers <= 1 or n < 8:
        for i, it in enumerate(items):
            results[i] = g(it)
            if desc and n >= 50 and (i + 1) % step == 0:
                log(f'      {desc} {i + 1}/{n}')
        return results
    ex = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'))
    broken = []
    try:
        futs = [ex.submit(g, it) for it in items]
        for i, f in enumerate(futs):
            try:
                results[i] = f.result()
            except BrokenProcessPool:
                broken.append(i)
            except Exception as e:                                   # noqa
                results[i] = Failure(f'{type(e).__name__}: {e}')
            if desc and n >= 50 and (i + 1) % step == 0:
                log(f'      {desc} {i + 1}/{n}')
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if broken:
        log(f'      ⚠ 进程池中途崩溃，对受影响的 {len(broken)} 项单独重试…')
        _isolate(g, items, broken, results)
    return results


def tb_text():
    return traceback.format_exc()


try:
    CFG
except NameError:
    CFG = {}

DEFAULTS = dict(
    # ---- 任务与输入 ----
    task='detect',                 # 'detect' 检测框 | 'segment' 实例分割
    img_dir='dataset/images',      # 图片文件夹（子文件夹也会递归读取，子文件夹名会被当作"分组"线索）
    ann_path='dataset/labels',     # 标注：txt/xml/mask 文件夹，或 COCO json 文件路径
    label_format='yolo',           # 'yolo' | 'coco' | 'voc' | 'yolo_seg' | 'mask'(语义:像素值=类别) | 'instmap'(实例:像素值=实例编号)
    classes_file=None,             # yolo/yolo_seg 的类别名文件（每行一个名字，或含 names: 的 yaml）；没有就用 class_names 或 class_0、class_1…
    class_names=None,              # 想固定类别顺序时填；None = 从数据里自动得到（COCO 取 categories，VOC 取标注里的名字）
    class_map={},                  # 合并/改名，如 {'plastic_bottle': 'bottle', 'can': 'bottle'}
    drop_classes=[],               # 直接忽略的类别名
    mask_values=None,              # mask 格式：{像素值: 类别名}；None 时 值1=第一个类别，依次类推
    inst_class='object',           # instmap 格式的类别名
    out_dir='prep_out',
    size=640,
    ratios=(0.8, 0.1, 0.1),        # train / val / test
    seed=0,
    workers=1,                     # 并行进程数（图多时调大，如 4）
    # ---- 严格检查 ----
    strict='skip',                 # 'abort' 有错误就停 | 'skip' 有错误的图隔离出去继续 | 'fix' 先自动修复能修的，修不了的隔离
    max_bad_frac=0.2,              # 被隔离的图超过这个比例就终止（多半是格式/路径配错了，不是个别坏数据）
    coord_tol=0.01,                # 坐标超出 0~1 不多于这个比例，当作标注工具的舍入误差，自动裁剪
    min_box_px=3,                  # 原图上宽或高小于这个像素的框丢掉（并记录）
    max_aspect=60,
    dup_iou=0.9,                   # 同类两个框 IoU 超过它视为重复，去掉一个
    exif='auto',                   # 'apply' 按 EXIF 旋转 | 'ignore' 不旋转 | 'auto' COCO/VOC 有尺寸时自动判断
    # ---- 分组与划分 ----
    group_mode='auto',             # 'auto' 多线索合并 | 'folder' | 'filename' | 'visual' | 'manifest' | 'none'
    group_manifest=None,           # csv：两列 image,group（image 写相对路径或文件名）
    visual_hash_thr=0.20,          # 视觉相似：256 位差值哈希里不同位的比例 ≤ 它
    visual_corr_thr=0.93,          # 或者：16x16 缩略图的相关系数 ≥ 它
    balance_split=True,            # 按类别均衡划分（按"组"为单位，保证同一组不跨集合）
    min_split_inst=3,              # val/test 里某类实例数少于它时警告
    # ---- 增强 ----
    copies=2,                      # 每张训练图的每种随机增强生成几份
    oversample=True,               # 稀有类别的图多生成几份
    oversample_max=4,              # 最多放大几倍
    proc_max_side=2560,            # 非切片变体先把大图缩到长边不超过它（更快、更省内存）；切片用原图
    tile_size=None, tile_overlap=0.2, tile_min_ratio=1.3, tile_keep_empty=0.1, tile_min_vis=0.4,
    degrees=10, translate=0.1, scale_lo=0.5, scale_hi=1.5, shear=2.0, perspective=0.0003,
    flip_lr=0.5, flip_ud=0.0,
    hsv=(0.015, 0.7, 0.4),         # 色调/饱和度/亮度的抖动幅度
    gamma=(0.7, 1.4), contrast=(0.8, 1.2), bright=(-20, 20),
    cls_margin=0.1, cls_size=224, cls_min_px=16,
    # ---- 分割专用 ----
    export_masks=True, min_inst_px=16, poly_eps=1.0,
    mask_nms_iou=0.8, max_area_frac=0.9, min_conf=0.0, keep_largest_part=False,
    missing_label='skip',          # 图没有标注文件时：'skip' 当作没标注、排除并警告 | 'background' 当作背景图（负样本）保留。（空的 txt 文件 = 明确的背景图，始终保留）
    max_group_frac=0.3,            # 自动分组后最大的一组不能超过总图数的这个比例（防止"相似的水面/叶子/废墟"被连成一个巨型组）
    rot90=False,                   # 随机增强里加 90° 倍数旋转（航拍俯视、叶片俯拍可开；对检测框无损）
    full_pix=['color'],            # aug_full 里随机叠加哪些像素增强：color / degrade / glare / haze / shadow / lowlight
    overwrite=False,               # 输出文件夹已存在：False 报错退出；True 只在它是本脚本生成的（有 reports/run_info.json）时才清空重建
    time_gap_sec=3.0,              # EXIF 拍摄时间相邻 ≤ 它（秒）的照片（连拍/同一航线）倾向归为一组；0 关闭。需画面也大致相似才采信
    max_fail_frac=0.1,             # 生成阶段处理失败的图超过这个比例就终止（失败的图会被整张清理掉，保证各变体一致）
    check_only=False,              # 只做读取、严格检查、分组、划分并写报告，不生成数据集（先跑这个看报告）
    variants=['base', 'clahe', 'aug_geo', 'aug_color', 'aug_full'],
)


def _cli(cfg):
    """命令行覆盖：python 脚本.py --img 图片夹 --ann 标注 --out 输出 --workers 4 --strict fix --check-only --overwrite
    放在模块顶层：Windows/macOS 的多进程子进程会重新导入脚本，必须拿到同样的配置"""
    ap = argparse.ArgumentParser(add_help=False)
    for k, t in (('img', str), ('ann', str), ('out', str), ('workers', int), ('strict', str), ('seed', int), ('format', str), ('classes', str), ('manifest', str), ('group', str), ('task', str), ('variants', str)):
        ap.add_argument('--' + k, type=t)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--help', '-h', action='store_true')
    ap.add_argument('--version', action='store_true')
    a, _ = ap.parse_known_args()
    if a.version:
        print('数据预处理 ' + __version__)
        raise SystemExit(0)
    if a.help:
        print(__doc__ or '见脚本顶部的 CFG 区')
        print('命令行：--variants base,tiles,aug_geo（逗号分隔） --img --ann --out --workers N --strict abort|skip|fix --format yolo|yolo_seg|coco|voc|mask|instmap --classes 类别文件 --seed --check-only --overwrite')
        raise SystemExit
    cfg = dict(cfg)
    for k, ck in (('img', 'img_dir'), ('ann', 'ann_path'), ('out', 'out_dir'), ('workers', 'workers'), ('strict', 'strict'), ('seed', 'seed'),
                  ('format', 'label_format'), ('classes', 'classes_file'), ('manifest', 'group_manifest'), ('group', 'group_mode'), ('task', 'task'), ('variants', 'variants')):
        if getattr(a, k) is not None:
            cfg[ck] = [x for x in getattr(a, k).split(',') if x] if k == 'variants' else getattr(a, k)
    if a.check_only:
        cfg['check_only'] = True
    if a.overwrite:
        cfg['overwrite'] = True
    return cfg


C = types.SimpleNamespace(**{**DEFAULTS, **_cli(CFG)})
TILE = C.tile_size or C.size
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
SEG = C.task == 'segment'


# ===================== 基础工具（imread_u 等在 p0_common）=====================
def base_of(name):
    """输出文件名 → 源图 stem（去掉 __k0 / __t3 后缀）"""
    return re.sub(r'__[kt]\d+$', '', name)


class Issues:
    """校验问题记录：level = ERROR(整张图隔离) / FIXED(已自动修复) / DROP(丢掉了某个标注) / WARN(提醒)"""
    def __init__(self):
        self.rows = []

    def add(self, level, code, key, detail='', line=''):
        self.rows.append((level, code, key, line, detail))

    def count(self, level=None, code=None):
        return sum(1 for r in self.rows if (level is None or r[0] == level) and (code is None or r[1] == code))

    def bad_keys(self):
        return {r[2] for r in self.rows if r[0] == 'ERROR'}

    def save(self, path):
        save_csv(path, ['level', 'code', 'image', 'line', 'detail'], self.rows)

    def summary(self):
        cnt = collections.Counter((r[0], r[1]) for r in self.rows)
        return cnt


# ===================== 1 图片发现 + 扫描（尺寸 / EXIF / 指纹）=====================
def discover_images(img_dir, issues):
    """递归找图。key = 相对路径去扩展名（'/' 分隔），同一个 key 出现两种扩展名（a.jpg 和 a.png）算错误"""
    files, seen = {}, {}
    for root, _, names in os.walk(img_dir, onerror=lambda e: issues.add('WARN', 'folder_unreadable', str(getattr(e, 'filename', '')), f'{type(e).__name__}: {e}')):
        for n in sorted(names, key=natural_key):
            if n.startswith('.') or n.startswith('._') or not n.lower().endswith(IMG_EXTS):   # 跳过 .DS_Store、Mac 的 ._ 隐藏文件
                continue
            full = os.path.join(root, n)
            key = os.path.splitext(os.path.relpath(full, img_dir))[0].replace(os.sep, '/')
            if key in seen:
                issues.add('ERROR', 'duplicate_stem', key, f'同名不同扩展名：{seen[key]} 和 {n}，标注无法区分')
                continue
            seen[key] = n
            files[key] = full
    return files


def scan_one(args):
    """扫描一张图（可在子进程里跑）：文件指纹、尺寸、EXIF 方向、用于分组的视觉哈希。不做完整解码，速度快"""
    key, path = args
    r = dict(key=key, ok=False)
    try:
        data = read_bytes(path)
        if data.size == 0:
            r['err'] = '空文件'
            return r
        r['md5'] = hashlib.md5(data).hexdigest()
        ori, raw, ts = 1, None, None
        if Image is not None:
            try:
                with Image.open(path) as im:
                    raw = im.size
                    ex = im.getexif()
                    ori = int(ex.get(274, 1) or 1)
                    t = ex.get_ifd(0x8769).get(36867) or ex.get(306)             # DateTimeOriginal / DateTime
                    if t:
                        import datetime as _dt
                        ts = _dt.datetime.strptime(str(t).strip()[:19], '%Y:%m:%d %H:%M:%S').timestamp()
            except Exception:                                                    # noqa
                raw = raw if raw else None
        c8 = cv.imdecode(data, cv.IMREAD_REDUCED_COLOR_8)               # 1/8 尺寸：JPEG 下解码很快
        g = cv.cvtColor(c8, cv.COLOR_BGR2GRAY) if c8 is not None and c8.size else None
        if g is None or g.size == 0:
            r['err'] = '无法解码（损坏、或格式如 HEIC 不被 OpenCV 支持）'
            return r
        if raw is None:                                                  # 没有 Pillow：只好完整解码一次拿尺寸
            full = cv.imdecode(data, cv.IMREAD_COLOR)
            if full is None:
                r['err'] = '无法解码'
                return r
            eff = (full.shape[1], full.shape[0])
            raw = eff
        else:
            eff = (raw[1], raw[0]) if ori in (5, 6, 7, 8) else raw
        r.update(ori=ori, raw=raw, eff=eff, ts=ts)
        s = cv.resize(g, (17, 16), interpolation=cv.INTER_AREA)
        r['dh'] = np.packbits((s[:, 1:] > s[:, :-1]).ravel())            # 256 位差值哈希
        t = cv.resize(g, (16, 16), interpolation=cv.INTER_AREA).astype(np.float32).ravel()
        r['th'] = ((t - t.mean()) / (t.std() + 1e-6)).astype(np.float32)  # 归一化缩略图（比相关系数）
        r['cm'] = c8.reshape(-1, 3).mean(0).astype(np.float32)           # 平均颜色：区分"结构都是平坦渐变"的不同场景
        r['ok'] = True
    except MemoryError:
        r['err'] = '内存不足（图片过大）'
    except Exception as e:                                               # noqa
        r['err'] = f'{type(e).__name__}: {e}'
    return r


# ===================== 2 标注读取：统一成 {图 key: [标注字典]} =====================
def read_names_file(path):
    """类别名文件：每行一个名字（obj.names / classes.txt），或 yaml 里的 names:（列表或 {0: a, 1: b}）"""
    if not path or not os.path.exists(path):
        return None
    txt, _enc = read_text(path)
    if path.lower().endswith(('.yaml', '.yml')):
        m = re.search(r'^names\s*:\s*(.*)$', txt, re.M)
        if m and m.group(1).strip().startswith('['):
            return [x.strip().strip('\'"') for x in m.group(1).strip()[1:-1].split(',') if x.strip()]
        names = {}
        for mm in re.finditer(r'^\s+(\d+)\s*:\s*(.+?)\s*$', txt[txt.find('names'):], re.M):
            names[int(mm.group(1))] = mm.group(2).strip('\'"')
        return [names[i] for i in sorted(names)] if names else None
    return [l.strip() for l in txt.splitlines() if l.strip()]


def find_label_file(key, index_full, index_base, ext):
    """优先按相对路径对应；对不上时按文件名找（文件名在标注里唯一才用）"""
    if key in index_full:
        return index_full[key]
    b = key.rsplit('/', 1)[-1]
    c = index_base.get(b, [])
    return c[0] if len(c) == 1 else None


def index_files(folder, ext):
    full, base = {}, collections.defaultdict(list)
    for root, _, names in os.walk(folder, onerror=lambda e: log(f'      ⚠ 文件夹读取失败：{e}')):
        for n in names:
            if n.endswith(ext) and not n.startswith('.'):
                p = os.path.join(root, n)
                k = os.path.splitext(os.path.relpath(p, folder))[0].replace(os.sep, '/')
                full[k] = p
                base[k.rsplit('/', 1)[-1]].append(p)
    return full, base


def ann(name, kind, geom, conf=1.0, line=''):
    return dict(name=name, kind=kind, geom=geom, conf=conf, line=line)


def parse_yolo_file(path, key, id2name, seg, issues):
    """YOLO txt → 标注列表。每一行都严格检查：列数、是否数字、有限值、类别编号是否为合法整数且在类别表内"""
    out = []
    try:
        txt, enc = read_text(path)
    except (OSError, PrepError) as e:
        issues.add('ERROR', 'label_unreadable', key, f'{type(e).__name__}: {e}')
        return out
    if enc != 'utf-8-sig':
        issues.add('WARN', 'encoding_fallback', key, f'标注文件不是 UTF-8，按 {enc} 读取')
    for ln, raw in enumerate(txt.splitlines(), 1):
        t = raw.split()
        if not t or t[0].startswith('#'):
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            issues.add('ERROR', 'nonnumeric', key, raw[:60], ln)
            continue
        if not all(np.isfinite(v)):
            issues.add('ERROR', 'nonfinite', key, raw[:60], ln)
            continue
        conf = 1.0
        if len(v) >= 8 and len(v) % 2 == 0 and seg:
            conf, v = v[-1], v[:-1]                       # 分割行尾多一个置信度
        elif len(v) == 6 and not seg:
            conf, v = v[-1], v[:-1]
            issues.add('WARN', 'has_conf', key, '检测行尾带置信度，已忽略', ln)
        cid = v[0]
        if cid != int(cid) or cid < 0:
            issues.add('ERROR', 'class_not_integer', key, f'类别 {t[0]}', ln)
            continue
        name = id2name(int(cid))
        if name is None:
            issues.add('ERROR', 'class_id_out_of_range', key, f'类别编号 {int(cid)} 不在类别表里', ln)
            continue
        if len(v) == 5:
            _, xc, yc, w, h = v
            if seg:
                out.append(ann(name, 'poly', np.array([[xc - w / 2, yc - h / 2], [xc + w / 2, yc - h / 2], [xc + w / 2, yc + h / 2], [xc - w / 2, yc + h / 2]]), conf, ln))
            else:
                out.append(ann(name, 'box', (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2), conf, ln))
        elif len(v) >= 7 and len(v) % 2 == 1:
            out.append(ann(name, 'poly', np.array(v[1:]).reshape(-1, 2), conf, ln))
        else:
            issues.add('ERROR', 'bad_columns', key, f'{len(v)} 列：{raw[:60]}', ln)
    return out


def load_coco_json(path):
    """读 COCO json；格式不对时给出明确的中文原因（而不是一长串 KeyError）"""
    try:
        txt, _ = read_text(path)
        d = json.loads(txt)
    except json.JSONDecodeError as e:
        raise InputError(f'COCO json 解析失败：第 {e.lineno} 行第 {e.colno} 列：{e.msg}', '文件可能没导完整/被截断，或不是 JSON；用编辑器打开检查末尾')
    except OSError as e:
        raise InputError(f'COCO json 读不了：{e}', os_error_hint(e))
    if not isinstance(d, dict):
        raise InputError('COCO json 顶层应该是对象 {…}', '确认导出格式选的是 "COCO 1.0"')
    miss = [k for k in ('images', 'annotations', 'categories') if k not in d or not isinstance(d[k], list)]
    if miss:
        raise InputError(f'COCO json 缺少 {miss}', '确认导出格式选的是 "COCO 1.0"；如果是 YOLO/VOC 格式请改 label_format')
    bad = [c for c in d['categories'] if not isinstance(c, dict) or 'id' not in c or 'name' not in c]
    if bad:
        raise InputError(f'categories 里有 {len(bad)} 个条目缺少 id/name', '检查导出')
    return d


def _coco_one(a, imgs, cname, declared, raw, issues, skipped):
    k = imgs.get(a['image_id'])
    if k is None:
        return
    nm = cname.get(a['category_id'])
    if nm is None:
        issues.add('ERROR', 'class_id_out_of_range', k, f"category_id {a['category_id']} 不在 categories 里")
        return
    if a.get('iscrowd', 0):
        skipped['iscrowd'] += 1
        return
    W, H = declared[k]
    if SEG:
        seg = a.get('segmentation')
        if isinstance(seg, list) and seg:
            for poly in seg:
                if len(poly) >= 6 and W and H:
                    raw[k].append(ann(nm, 'poly', np.array(poly, float).reshape(-1, 2) / [W, H], 1.0, f"ann {a.get('id')}"))
                else:
                    skipped['polygon 太短'] += 1
        elif a.get('bbox') and W and H:
            x, y, w, h = a['bbox']
            raw[k].append(ann(nm, 'poly', np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) / [W, H], 1.0, f"ann {a.get('id')}"))
            skipped['无 segmentation，用 bbox 代替(矩形)'] += 1
        else:
            skipped['RLE 或空'] += 1
    else:
        if a.get('bbox') and W and H:
            x, y, w, h = a['bbox']
            raw[k].append(ann(nm, 'box', (x / W, y / H, (x + w) / W, (y + h) / H), 1.0, f"ann {a.get('id')}"))
        else:
            skipped['无 bbox'] += 1


def load_annotations(img_keys, issues):
    """返回 (标注字典, 有标注文件的 key 集合, 声明的图像尺寸字典, 源类别顺序, 标注方式 'norm'/'px')"""
    fmt, raw, has_file, declared, src_order = C.label_format, {}, set(), {}, None
    keys = set(img_keys)
    if fmt in ('yolo', 'yolo_seg'):
        seg = (fmt == 'yolo_seg')
        names = read_names_file(C.classes_file) or C.class_names
        if C.classes_file and not names:
            issues.add('WARN', 'classes_file_unreadable', C.classes_file, '读不出类别名，改用 class_names 或 class_N')
        if names:
            id2name = lambda i: names[i] if i < len(names) else None
        else:
            id2name = lambda i: f'class_{i}'
        src_order = list(names) if names else None
        full, base = index_files(C.ann_path, '.txt')
        for k in img_keys:
            p = find_label_file(k, full, base, '.txt')
            if p is None:
                continue
            has_file.add(k)
            raw[k] = parse_yolo_file(p, k, id2name, seg, issues)
        for k in full:
            if k not in keys and k.rsplit('/', 1)[-1] not in {x.rsplit('/', 1)[-1] for x in keys} and k not in ('classes', 'obj'):
                issues.add('WARN', 'label_without_image', k, '有标注文件但找不到对应的图')
    elif fmt == 'coco':
        d = load_coco_json(C.ann_path)
        cats = sorted(d['categories'], key=lambda c: c['id'])
        cname = {c['id']: c['name'] for c in cats}
        src_order = [c['name'] for c in cats]        # 类别自动取自 COCO 的 categories，不再依赖手写的类别表
        by_name = {}
        for k in img_keys:
            by_name.setdefault(k, k)
            by_name.setdefault(k.rsplit('/', 1)[-1], k)
        imgs = {}
        for im in d['images']:
            if not isinstance(im, dict) or 'id' not in im or 'file_name' not in im:
                issues.add('ERROR', 'coco_bad_image_entry', '(coco)', f'images 里有缺少 id/file_name 的条目：{str(im)[:80]}')
                continue
            fn = os.path.splitext(str(im['file_name']).replace('\\', '/'))[0]
            k = by_name.get(fn) or by_name.get(fn.rsplit('/', 1)[-1])
            if k is None:
                issues.add('WARN', 'coco_image_not_found', fn, 'json 里有这张图，但图片文件夹里没有')
                continue
            imgs[im['id']] = k
            has_file.add(k)
            raw.setdefault(k, [])
            declared[k] = (im.get('width'), im.get('height'))
        skipped = collections.Counter()
        for a in d['annotations']:
            try:
                _coco_one(a, imgs, cname, declared, raw, issues, skipped)
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
                issues.add('ERROR', 'coco_bad_annotation', imgs.get(a.get('image_id') if isinstance(a, dict) else None, '(coco)'), f'标注 {a.get("id") if isinstance(a, dict) else "?"} 结构不对：{type(e).__name__} {e}')
        for why, n in skipped.items():
            issues.add('WARN', 'coco_skipped', '(全局)', f'{why}：{n} 个')
    elif fmt == 'voc':
        full, base = index_files(C.ann_path, '.xml')
        names_seen = set()
        for k in img_keys:
            p = find_label_file(k, full, base, '.xml')
            if p is None:
                continue
            has_file.add(k)
            raw[k] = []
            try:
                root = ET.parse(p).getroot()
                sz = root.find('size')
                W, H = int(float(sz.find('width').text)), int(float(sz.find('height').text))
                declared[k] = (W, H)
                for o in root.findall('object'):
                    nm = o.find('name').text.strip()
                    names_seen.add(nm)
                    bb = o.find('bndbox')
                    x1, y1, x2, y2 = [float(bb.find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax')]
                    raw[k].append(ann(nm, 'box', ((x1 - 1) / W, (y1 - 1) / H, x2 / W, y2 / H), 1.0))   # VOC 坐标是 1 起始
            except Exception as e:                                                                   # noqa
                issues.add('ERROR', 'xml_parse_error', k, f'{type(e).__name__}: {e}')
        src_order = sorted(names_seen)
    elif fmt in ('mask', 'instmap'):
        full, base = index_files(C.ann_path, '.png')
        for k in img_keys:
            p = find_label_file(k, full, base, '.png')
            if p is not None:
                has_file.add(k)
                raw[k] = [ann(None, 'maskfile', p)]
        src_order = None
    else:
        raise ConfigError(f"label_format 不认识：{fmt!r}", "只能是 'yolo' / 'yolo_seg' / 'coco' / 'voc' / 'mask' / 'instmap'")
    return raw, has_file, declared, src_order


if C.label_format == 'mask' and not C.class_names and C.classes_file:      # mask 格式也能用类别名文件（模块顶层执行：多进程子进程重新导入时同样生效）
    C.class_names = read_names_file(C.classes_file)

# ===================== 3 严格校验 + 类别解析 =====================
def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def mask_task(args):
    """mask / instmap 标注图 → 多边形（在子进程里跑）。返回 (key, [(类别名, 归一化多边形)], [(level, code, detail)])"""
    key, path, (w, h) = args
    probs, out = [], []
    mk = imread_u(path, cv.IMREAD_UNCHANGED)
    if mk is None:
        return key, [], [('ERROR', 'mask_unreadable', '标注 png 打不开')]
    if mk.ndim == 3:
        if mk.shape[2] >= 3 and not (np.array_equal(mk[:, :, 0], mk[:, :, 1]) and np.array_equal(mk[:, :, 1], mk[:, :, 2])):
            return key, [], [('ERROR', 'mask_is_color', '标注是彩色图，需要"单通道、像素值=类别/实例编号"的索引图（SAM 导出时选 index/label 格式，或自己把颜色换成编号）')]
        mk = mk[:, :, 0]
    if mk.dtype != np.uint8:
        if mk.max() > 255:
            if C.label_format == 'instmap' and mk.max() < 65536:
                pass                                   # 16 位实例图可以
            else:
                return key, [], [('ERROR', 'mask_range', f'像素值最大 {int(mk.max())}，超过 255')]
    if mk.shape != (h, w):
        return key, [], [('ERROR', 'mask_size_mismatch', f'标注 {mk.shape[::-1]} 和图 {(w, h)} 尺寸不一致（常见原因：图带 EXIF 旋转、或标注是在缩放后的图上做的）')]
    k8 = np.ones((3, 3), np.uint8)

    def emit(m, name):
        if C.keep_largest_part:
            n, lab, st, _ = cv.connectedComponentsWithStats(m, connectivity=8)
            if n > 2:
                m = (lab == 1 + int(np.argmax(st[1:, cv.CC_STAT_AREA]))).astype(np.uint8)
        cs, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for ct in cs:
            if cv.contourArea(ct) + len(ct) / 2 < C.min_inst_px:
                probs.append(('DROP', 'tiny_poly', f'{name} 面积过小'))
                continue
            ap = cv.approxPolyDP(ct, 0.5, True).reshape(-1, 2)
            if len(ap) >= 3:
                out.append((name, ((ap + 0.5) / [w, h]).clip(0, 1)))
    vals = [int(v) for v in np.unique(mk) if v != 0]
    for v in vals:
        m = (mk == v).astype(np.uint8)
        if C.label_format == 'instmap':
            emit(m, C.inst_class)
        else:
            name = (C.mask_values or {}).get(v) if C.mask_values else (C.class_names[v - 1] if C.class_names and 0 < v <= len(C.class_names) else f'class_{v - 1}')
            if name is None:
                probs.append(('ERROR', 'mask_unknown_value', f'像素值 {v} 不在 mask_values 里'))
                continue
            n, lab = cv.connectedComponents(m, connectivity=8)       # 语义图没有实例信息：每个连通块当一个目标
            for i in range(1, n):
                emit((lab == i).astype(np.uint8), name)
    return key, out, probs


def clean_polys(anns, w, h, key, issues):
    """SAM / 自动标注常见问题：置信度低、几乎铺满整张图的"背景"、同一目标被重复分割。在缩小的 mask 上判断（快）"""
    if not anns:
        return anns
    s = min(1.0, 384 / max(w, h))
    sw, sh = max(1, int(w * s)), max(1, int(h * s))
    ms, keep = [], []
    for a in anns:
        if C.min_conf > 0 and a['conf'] < C.min_conf:
            issues.add('DROP', 'low_conf', key, f"{a['name']} 置信度 {a['conf']:.2f} < {C.min_conf}")
            continue
        m = np.zeros((sh, sw), np.uint8)
        cv.fillPoly(m, [np.round(a['geom'] * [sw, sh] * 16).astype(np.int32)], 1, shift=4)
        if m.sum() > C.max_area_frac * sw * sh:
            issues.add('DROP', 'near_full_image', key, f"{a['name']} 面积占整图 {m.sum() / (sw * sh):.0%}，多半是背景")
            continue
        ms.append(m.astype(bool))
        keep.append(a)
    order = sorted(range(len(keep)), key=lambda i: (-keep[i]['conf'], -int(ms[i].sum())))
    out, kept_m = [], []
    for i in order:
        dup = False
        for j in range(len(out)):
            if out[j]['name'] == keep[i]['name']:
                inter = np.logical_and(ms[i], kept_m[j]).sum()
                if inter / max(1, np.logical_or(ms[i], kept_m[j]).sum()) > C.mask_nms_iou:
                    dup = True
                    break
        if dup:
            issues.add('DROP', 'duplicate_mask', key, f"{keep[i]['name']} 与已有 mask 重叠 IoU>{C.mask_nms_iou}")
        else:
            out.append(keep[i])
            kept_m.append(ms[i])
    return out


def check_anns(anns, w, h, key, issues):
    """逐个标注严格检查；返回清洗后的列表。有 ERROR 记入 issues（整张图之后会被隔离）。
    坐标超界：≤coord_tol 当舍入误差直接裁剪；更大的在 strict='fix' 时裁剪，否则 ERROR（多半是标注/图对不上）"""
    out = []
    for a in anns:
        g = a['geom']
        if a['kind'] == 'box':
            x1, y1, x2, y2 = [float(t) for t in g]
            if x2 < x1 or y2 < y1:
                issues.add('ERROR', 'negative_size', key, f'宽或高为负：{a["line"]}', a['line'])
                continue
            vals = np.array([x1, y1, x2, y2])
            ex = max(0.0, -vals.min(), vals.max() - 1)
            if ex > 0:
                if ex <= C.coord_tol or C.strict == 'fix':
                    issues.add('FIXED', 'coord_clipped', key, f'超界 {ex:.4f}，已裁剪到 0~1', a['line'])
                    x1, y1, x2, y2 = np.clip(vals, 0, 1)
                else:
                    issues.add('ERROR', 'coord_out_of_range', key, f'坐标超出 0~1 达 {ex:.3f}（超过容差 {C.coord_tol}）：标注和图对不上？', a['line'])
                    continue
            bw, bh = (x2 - x1) * w, (y2 - y1) * h
            if bw <= 0 or bh <= 0:
                issues.add('DROP', 'zero_size', key, '宽或高为 0', a['line'])
                continue
            if bw < C.min_box_px or bh < C.min_box_px:
                issues.add('DROP', 'tiny_box', key, f'{a["name"]} 原图上仅 {bw:.1f}×{bh:.1f}px（< {C.min_box_px}）', a['line'])
                continue
            if max(bw / bh, bh / bw) > C.max_aspect:
                issues.add('WARN', 'extreme_aspect', key, f'{a["name"]} 宽高比 {max(bw / bh, bh / bw):.0f}', a['line'])
            a = dict(a, geom=(x1, y1, x2, y2))
        else:
            p = np.asarray(g, float)
            keep = np.ones(len(p), bool)
            keep[1:] = np.any(np.abs(np.diff(p, axis=0)) > 1e-9, axis=1)      # 去掉连续重复点
            p = p[keep]
            ex = max(0.0, float(-p.min()), float(p.max() - 1))
            if ex > 0:
                if ex <= C.coord_tol or C.strict == 'fix':
                    issues.add('FIXED', 'coord_clipped', key, f'超界 {ex:.4f}，已裁剪', a['line'])
                    p = p.clip(0, 1)
                else:
                    issues.add('ERROR', 'coord_out_of_range', key, f'多边形坐标超出 0~1 达 {ex:.3f}', a['line'])
                    continue
            if len(p) < 3 or poly_area(p * [w, h]) < C.min_inst_px:
                issues.add('DROP', 'tiny_poly', key, f'{a["name"]} 点数不足或面积 < {C.min_inst_px}px', a['line'])
                continue
            a = dict(a, geom=p)
        out.append(a)
    return out


def dedup_boxes(anns, key, issues):
    out = []
    for a in anns:
        dup = False
        if a['kind'] == 'box':
            for b in out:
                if b['name'] == a['name'] and b['kind'] == 'box':
                    ix = max(0, min(a['geom'][2], b['geom'][2]) - max(a['geom'][0], b['geom'][0]))
                    iy = max(0, min(a['geom'][3], b['geom'][3]) - max(a['geom'][1], b['geom'][1]))
                    ua = (a['geom'][2] - a['geom'][0]) * (a['geom'][3] - a['geom'][1]) + (b['geom'][2] - b['geom'][0]) * (b['geom'][3] - b['geom'][1]) - ix * iy
                    if ix * iy / max(ua, 1e-12) > C.dup_iou:
                        dup = True
                        break
        if dup:
            issues.add('DROP', 'duplicate_box', key, f'{a["name"]} 与已有框 IoU>{C.dup_iou}', a['line'])
        else:
            out.append(a)
    return out


def build_samples(files, scans, issues):
    """把"图 + 标注"整理成统一的样本表。返回 (samples 字典 key→样本, 类别名列表, 类别映射表行)"""
    raw, has_file, declared, src_order = load_annotations(list(files), issues)
    # ---- 掩码类标注：并行转成多边形 ----
    mask_jobs = []
    samples, cmap = {}, {}
    for k, path in files.items():
        r = scans.get(k)
        if r is None or not r['ok']:
            issues.add('ERROR', 'unreadable_image', k, (r or {}).get('err', '未知错误'))
            continue
        raw_size, eff, ori = r['raw'], r['eff'], r['ori']
        ignore_exif = False
        if ori != 1:
            d = declared.get(k)
            if d and d[0]:
                if tuple(d) == tuple(eff):
                    pass
                elif tuple(d) == tuple(raw_size):
                    ignore_exif, eff = True, raw_size
                    issues.add('FIXED', 'exif_ignored', k, f'标注尺寸 {d} 对应未旋转的原始像素，按它处理（不应用 EXIF 旋转）')
            elif C.exif == 'ignore':
                ignore_exif, eff = True, raw_size
        d = declared.get(k)
        if d and d[0] and tuple(d) != tuple(eff):
            issues.add('ERROR', 'size_mismatch', k, f'标注文件里写的尺寸 {tuple(d)} 和真实图片 {tuple(eff)} 不一致，所有坐标都会错位')
            continue
        if k not in has_file:
            if C.missing_label == 'background':
                issues.add('WARN', 'missing_label_as_background', k, '没有标注文件，按背景图保留')
                raw[k] = []
            else:
                issues.add('WARN', 'missing_label_skipped', k, '没有标注文件：不知道是背景还是漏标，已排除（想当背景图用请设 missing_label="background"）')
                continue
        samples[k] = dict(key=k, path=path, eff=tuple(eff), ignore_exif=ignore_exif, md5=r['md5'], ori=ori, anns=raw.get(k, []))
        if C.label_format in ('mask', 'instmap') and raw.get(k):
            mask_jobs.append((k, raw[k][0]['geom'], tuple(eff)))
    if mask_jobs:
        for job, res in zip(mask_jobs, run_pool(mask_task, mask_jobs, C.workers, '标注图转多边形')):
            k = job[0]
            if isinstance(res, Failure):
                issues.add('ERROR', 'mask_failed', k, res.msg + ('；' + res.tb.splitlines()[-1] if res.tb else ''))
                samples[k]['anns'] = []
                continue
            _, polys, probs = res
            for lv, code, det in probs:
                issues.add(lv, code, k, det)
            samples[k]['anns'] = [ann(n, 'poly', p) for n, p in polys]
        names_seen = collections.OrderedDict()
        for s in samples.values():
            for a in s['anns']:
                names_seen[a['name']] = 1
        src_order = (list(C.class_names) if C.class_names else None) or sorted(names_seen)
    # ---- 类别映射 ----
    drop = set(C.drop_classes)
    def final_name(n):
        if n in drop:
            return None
        return C.class_map.get(n, n)
    explicit = list(C.class_names) if (C.class_names and C.label_format not in ('yolo', 'yolo_seg', 'mask')) else None
    if C.class_names and C.label_format in ('yolo', 'yolo_seg', 'mask'):
        explicit = [C.class_map.get(n, n) for n in C.class_names if n not in drop]
        explicit = list(collections.OrderedDict((n, 1) for n in explicit))
    # ---- 逐图清洗 ----
    for k in list(samples):
        s = samples[k]
        w, h = s['eff']
        anns = []
        for a in s['anns']:
            n = final_name(a['name'])
            if n is None:
                continue
            a = dict(a, name=n)
            if explicit is not None and n not in explicit:
                issues.add('ERROR', 'unknown_class', k, f'类别 "{n}" 不在 class_names 里（也没被 class_map 映射到里面）', a['line'])
                continue
            if not SEG and a['kind'] == 'poly':          # 检测任务读到多边形：取外接框
                p = np.asarray(a['geom'])
                a = dict(a, kind='box', geom=(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
            elif SEG and a['kind'] == 'box':
                x1, y1, x2, y2 = a['geom']
                a = dict(a, kind='poly', geom=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
                issues.add('WARN', 'box_used_as_polygon', k, '分割任务里读到的是框，按矩形多边形处理')
            anns.append(a)
        anns = check_anns(anns, w, h, k, issues)
        anns = clean_polys(anns, w, h, k, issues) if SEG else dedup_boxes(anns, k, issues)
        s['anns'] = anns
    # ---- 类别编号：自动、连续、顺序稳定 ----
    used = collections.Counter(a['name'] for s in samples.values() for a in s['anns'])
    if explicit is not None:
        names = list(explicit)
    else:
        order = []
        for n in (src_order or []):
            fn = final_name(n)
            if fn is not None and fn not in order:
                order.append(fn)
        for n in sorted(used):
            if n not in order:
                order.append(n)
        names = [n for n in order if used.get(n, 0) > 0]
        gone = [n for n in order if used.get(n, 0) == 0]
        if gone:
            issues.add('WARN', 'class_without_instances', '(全局)', f'这些类别在数据里没有任何有效实例，已不放进类别表：{gone}')
    id_of = {n: i for i, n in enumerate(names)}
    for s in samples.values():
        for a in s['anns']:
            a['cid'] = id_of[a['name']]
    srcs = collections.defaultdict(list)
    for n in (src_order or sorted({a['name'] for s in samples.values() for a in s['anns']})):
        fn = final_name(n)
        srcs[fn].append(n)
    cmap_rows = [(i, n, ' + '.join(srcs.get(n, [n])), used.get(n, 0)) for i, n in enumerate(names)]
    return samples, names, cmap_rows


# ===================== 4 分组（防泄漏）+ 按类别均衡划分 + 过采样倍数 =====================
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)
            return True
        return False


def visual_edges(DH, TH, CM, scale):
    """两两比较（分块矩阵乘法）：256 位差值哈希的不同位比例 ≤ thr，或 16x16 缩略图相关系数 ≥ corr_thr"""
    n = len(DH)
    B = np.unpackbits(DH, axis=1).astype(np.float32)
    ham_t = C.visual_hash_thr * scale * 256
    corr_t = 1 - (1 - C.visual_corr_thr) * scale
    ei, ej = [], []
    blk = 512
    for i0 in range(0, n, blk):
        i1 = min(n, i0 + blk)
        ham = 256 - (B[i0:i1] @ B.T + (1 - B[i0:i1]) @ (1 - B).T)
        cor = TH[i0:i1] @ TH.T / TH.shape[1]
        cd = np.abs(CM[i0:i1, None, :] - CM[None, :, :]).max(2) <= 30        # 平均颜色差太大不可能是同一段视频
        m = ((ham <= ham_t) | (cor >= corr_t)) & cd
        ii, jj = np.nonzero(m)
        sel = jj > ii + i0
        ei.append(ii[sel] + i0)
        ej.append(jj[sel])
        if sum(len(x) for x in ei) > 3_000_000:
            return None
    return np.concatenate(ei), np.concatenate(ej)


def make_groups(samples, scans, names, issues):
    """多线索合并成"组"，同一组的图一定进同一个集合（视频帧、连拍、重复图都不能跨 train/val）：
    ① 完全相同的文件（md5） ② group_manifest 里手填的分组 ③ 子文件夹（视频一个文件夹） ④ 文件名里的帧序号——只在相邻帧"看起来也像"时才信
    ⑤ 视觉相似（差值哈希/缩略图），文件名完全乱掉的视频抽帧也能连起来。线索 ④⑤ 会防止"一个巨型组"。"""
    keys = sorted(samples, key=natural_key)
    n = len(keys)
    idx = {k: i for i, k in enumerate(keys)}
    uf = UF(n)
    why = collections.Counter()
    mode = C.group_mode
    if mode == 'none' or n == 0:
        return {k: k for k in keys}, why
    # ① 完全重复
    bymd5 = collections.defaultdict(list)
    for k in keys:
        bymd5[samples[k]['md5']].append(k)
    for ks in bymd5.values():
        for k in ks[1:]:
            if uf.union(idx[ks[0]], idx[k]):
                why['exact_duplicate'] += 1
    if mode in ('auto', 'manifest') and C.group_manifest:
        if not os.path.exists(C.group_manifest):
            raise InputError('group_manifest 文件不存在：' + C.group_manifest, '检查路径；它由"标注前"脚本生成（pre_out/group_manifest.csv）')
        gm = collections.defaultdict(list)
        base = collections.defaultdict(list)
        for k in keys:
            base[k.rsplit('/', 1)[-1]].append(k)
        for row in csv.reader(read_text(C.group_manifest)[0].splitlines()):
            if len(row) < 2 or row[0].strip().lower() in ('image', 'file', 'filename'):
                continue
            im = os.path.splitext(row[0].strip().replace('\\', '/'))[0]
            k = im if im in idx else (base[im.rsplit('/', 1)[-1]][0] if len(base.get(im.rsplit('/', 1)[-1], [])) == 1 else None)
            if k is None:
                issues.add('WARN', 'manifest_image_not_found', row[0], 'group_manifest 里的图找不到（或文件名重名）')
                continue
            gm[row[1].strip()].append(k)
        for ks in gm.values():
            for k in ks[1:]:
                if uf.union(idx[ks[0]], idx[k]):
                    why['manifest'] += 1
    if mode in ('auto', 'folder'):
        cls_l = {c.lower() for c in names}
        fold = collections.defaultdict(list)
        for k in keys:
            if '/' in k:
                fold[k.rsplit('/', 1)[0]].append(k)
        ign = [f for f in fold if f.rsplit('/', 1)[-1].lower() in cls_l]
        if ign:
            issues.add('WARN', 'class_folders_ignored', '(全局)', f'这些文件夹名和类别名相同，当成"按类别放图"而不是"按视频放图"，不用于分组：{ign[:5]}')
        usable = {f: ks for f, ks in fold.items() if f not in ign}
        if len(usable) >= 2 or mode == 'folder':
            for ks in usable.values():
                for k in ks[1:]:
                    if uf.union(idx[ks[0]], idx[k]):
                        why['folder'] += 1
    TH = DH = CM = None
    ok_vis = [k for k in keys if scans.get(k, {}).get('ok')]
    if ok_vis:
        DH = np.stack([scans[k]['dh'] for k in keys if k in scans and scans[k].get('ok')])
        TH = np.stack([scans[k]['th'] for k in keys if k in scans and scans[k].get('ok')])
        CM = np.stack([scans[k]['cm'] for k in keys if k in scans and scans[k].get('ok')])
    vidx = {k: i for i, k in enumerate(ok_vis)}
    if mode in ('auto', 'filename'):
        pre = collections.defaultdict(list)
        for k in keys:
            m = re.match(r'^(.*?)[_\-\s.]*(\d+)$', k.rsplit('/', 1)[-1])
            if m and m.group(1):
                pre[(k.rsplit('/', 1)[0] if '/' in k else '', m.group(1))].append((int(m.group(2)), k))
        for lst in pre.values():
            lst.sort()
            for (_, a), (_, b) in zip(lst, lst[1:]):
                if mode == 'filename':
                    good = True
                elif a in vidx and b in vidx:
                    good = float(TH[vidx[a]] @ TH[vidx[b]] / TH.shape[1]) >= 0.5       # 文件名说是相邻帧，画面也得大致相似才信
                else:
                    good = False
                if good and uf.union(idx[a], idx[b]):
                    why['filename'] += 1
    if mode in ('auto', 'filename', 'visual') and C.time_gap_sec > 0:
        tk = sorted((scans[k]['ts'], k) for k in keys if scans.get(k, {}).get('ts'))
        for (t1, a), (t2, b) in zip(tk, tk[1:]):
            if t2 - t1 <= C.time_gap_sec:
                good = a in vidx and b in vidx and float(TH[vidx[a]] @ TH[vidx[b]] / TH.shape[1]) >= 0.3     # 拍摄时间相邻，画面也别差太远（防止同一秒拍的两个不同地点）
                if good and uf.union(idx[a], idx[b]):
                    why['exif_time'] += 1
    if mode in ('auto', 'visual') and DH is not None and len(ok_vis) > 1:
        scale, edges = 1.0, None
        while edges is None and scale > 0.05:              # 边太多（图都很像）时先收紧阈值，避免内存爆炸
            edges = visual_edges(DH, TH, CM, scale)
            if edges is None:
                scale *= 0.5
        if edges is not None and len(edges[0]):
            ei, ej = edges
            B = np.unpackbits(DH, axis=1)
            ham = (B[ei] != B[ej]).sum(axis=1)
            cor = (TH[ei] * TH[ej]).sum(axis=1) / TH.shape[1]
            score = cor - ham / 256.0                          # 越大越像
            cap = max(2, int(C.max_group_frac * n))
            size = collections.Counter(uf.find(i) for i in range(n))
            refused = 0
            for e in np.argsort(-score, kind='stable'):        # 从最像的连接开始合并：同一段视频里相邻帧最像，先连好；跨场景的弱连接最后才轮到
                ra, rb = uf.find(idx[ok_vis[ei[e]]]), uf.find(idx[ok_vis[ej[e]]])
                if ra == rb:
                    continue
                if size[ra] + size[rb] > cap:                  # 合并后会超过上限：不合并（防止相似的水面/叶子被连成巨型组）
                    refused += 1
                    continue
                uf.union(ra, rb)
                size[min(ra, rb)] = size[ra] + size[rb]
                why['visual'] += 1
            if refused:
                issues.add('WARN', 'group_size_capped', '(全局)', f'有 {refused} 条"看起来相似"的连接因为会让一个组超过 {cap} 张（{C.max_group_frac:.0%}）而没合并：图片整体很相似，分组可能不够完整，建议用文件夹或 group_manifest 明确分组')
    sizes = collections.Counter(uf.find(i) for i in range(n))
    if sizes and max(sizes.values()) > C.max_group_frac * n and n >= 10:
        issues.add('WARN', 'giant_group', '(全局)', f'最大的组有 {max(sizes.values())} 张（{max(sizes.values()) / n:.0%}）：来自文件夹/清单/时间线索的合并本身就很大，划分会不均匀')
    comp = collections.defaultdict(list)
    for k in keys:
        comp[uf.find(idx[k])].append(k)
    groups = {}
    for ks in comp.values():
        for k in ks:
            groups[k] = ks[0]
    return groups, why


def split_groups(samples, groups, names, issues):
    """以"组"为单位，按类别均衡地分 train/val/test：
    目标 = 每个集合里每个类别的实例数 ≈ 比例×该类总数，图片数 ≈ 比例×总图数。先贪心放（稀有类别多的组先放），再随机爬山优化（移动/互换组）。
    各集合里每个类别（总数够多时）至少要有实例。"""
    split_names = ['train', 'val', 'test']
    r = np.array(C.ratios, float)
    r = r / r.sum()
    act = [i for i in range(3) if r[i] > 0]
    gk = sorted(set(groups.values()), key=natural_key)
    gi = {g: i for i, g in enumerate(gk)}
    G, K = len(gk), len(names)
    cnt = np.zeros((G, K))
    nim = np.zeros(G)
    for k, s in samples.items():
        g = gi[groups[k]]
        nim[g] += 1
        for a in s['anns']:
            cnt[g, a['cid']] += 1
    T, N = cnt.sum(0), nim.sum()
    rnd = random.Random(C.seed)
    if len(act) > G:
        issues.add('WARN', 'too_few_groups', '(全局)', f'只有 {G} 个组，不够分给 {len(act)} 个集合')
    def cost(sc, si):
        c = 0.0
        for s in act:
            tgt = r[s] * T
            c += float((((sc[s] - tgt) / np.maximum(T, 1)) ** 2).sum()) / max(K, 1)
            c += ((si[s] - r[s] * N) / max(N, 1)) ** 2
            if si[s] == 0:
                c += 5.0
            c += 1.0 * float(((sc[s] == 0) & (r[s] * T >= 1)).sum())      # 每个集合都不能缺类别（总数够分的前提下）
        return c
    rare = (cnt / np.maximum(T, 1)).sum(1)
    order = sorted(range(G), key=lambda g: (-rare[g], -nim[g], rnd.random()))
    asg = np.zeros(G, int)
    sc = np.zeros((3, K)); si = np.zeros(3)
    for g in order:
        best, bs = None, None
        for s in act:
            sc[s] += cnt[g]; si[s] += nim[g]
            c = cost(sc, si)
            sc[s] -= cnt[g]; si[s] -= nim[g]
            if best is None or c < best:
                best, bs = c, s
        asg[g] = bs
        sc[bs] += cnt[g]; si[bs] += nim[g]
    cur = cost(sc, si)
    for it in range(min(30000, 300 * G)):
        g = rnd.randrange(G)
        s0 = asg[g]
        if rnd.random() < 0.5 and len(act) > 1:
            s1 = rnd.choice([s for s in act if s != s0])
            sc[s0] -= cnt[g]; si[s0] -= nim[g]; sc[s1] += cnt[g]; si[s1] += nim[g]
            c = cost(sc, si)
            if c < cur - 1e-12:
                cur, asg[g] = c, s1
            else:
                sc[s0] += cnt[g]; si[s0] += nim[g]; sc[s1] -= cnt[g]; si[s1] -= nim[g]
        else:
            h = rnd.randrange(G)
            s1 = asg[h]
            if s1 == s0:
                continue
            d = cnt[h] - cnt[g]; dn = nim[h] - nim[g]
            sc[s0] += d; si[s0] += dn; sc[s1] -= d; si[s1] -= dn
            c = cost(sc, si)
            if c < cur - 1e-12:
                cur, asg[g], asg[h] = c, s1, s0
            else:
                sc[s0] -= d; si[s0] -= dn; sc[s1] += d; si[s1] += dn
    out = {k: split_names[asg[gi[groups[k]]]] for k in samples}
    return out


def split_report(samples, assign, names, issues):
    rows = []
    n_img = collections.Counter(assign.values())
    ins = collections.defaultdict(collections.Counter)
    imc = collections.defaultdict(collections.Counter)
    for k, s in samples.items():
        sp = assign[k]
        for c in {a['cid'] for a in s['anns']}:
            imc[sp][c] += 1
        for a in s['anns']:
            ins[sp][a['cid']] += 1
    for i, n in enumerate(names):
        tot = sum(ins[sp][i] for sp in ('train', 'val', 'test'))
        for sp in ('train', 'val', 'test'):
            rows.append((n, sp, ins[sp][i], imc[sp][i], f'{ins[sp][i] / max(tot, 1):.3f}'))
        for sp in ('val', 'test'):
            if C.ratios[1 if sp == 'val' else 2] > 0 and ins[sp][i] < C.min_split_inst:
                issues.add('WARN', 'few_instances_in_split', '(全局)', f'类别 "{n}" 在 {sp} 里只有 {ins[sp][i]} 个实例（< {C.min_split_inst}）：该类的验证指标不可信，需要补数据')
    return rows, n_img


def repeat_factors(samples, assign, names):
    """过采样倍数（LVIS 的 repeat-factor 思路）：f_c = 含类别 c 的训练图占比；r_c = max(1, sqrt(t/f_c))，图的倍数 = 它含有的类别里最大的 r_c，封顶 oversample_max。
    t 取各类 f_c 的中位数：中位数及以上的类别不放大，更稀有的才放大。平衡数据集上全是 1.0。"""
    tr = [s for k, s in samples.items() if assign[k] == 'train']
    if not tr or not C.oversample or not names:
        return {s['key']: 1.0 for s in tr}
    f = np.zeros(len(names))
    for s in tr:
        for c in {a['cid'] for a in s['anns']}:
            f[c] += 1
    f = f / len(tr)
    t = float(np.median(f[f > 0])) if (f > 0).any() else 1.0
    rc = np.where(f > 0, np.maximum(1.0, np.sqrt(t / np.maximum(f, 1e-9))), 1.0)
    rc = np.minimum(rc, C.oversample_max)
    return {s['key']: (max(rc[c] for c in {a['cid'] for a in s['anns']}) if s['anns'] else 1.0) for s in tr}


# ===================== 5 几何算子（检测框 / 分割 mask 统一接口）=====================
# items = [(类别编号, 几何)]；检测：几何 = [x1,y1,x2,y2]（像素）；分割：几何 = uint8 mask（和图同大小）
MIN_VIS = 0.2


def to_items(anns, w, h):
    items = []
    for a in anns:
        if SEG:
            m = np.zeros((h, w), np.uint8)
            cv.fillPoly(m, [np.round(np.asarray(a['geom']) * [w, h] * 16).astype(np.int32)], 1, shift=4)
            if int(m.sum()) >= max(1, C.min_inst_px // 4):
                items.append((a['cid'], m))
        else:
            items.append((a['cid'], np.array(a['geom'], np.float64) * [w, h, w, h]))
    return items


def lb(img, items, size, color=114):
    """letterbox：等比缩放 + 居中补边。保持长宽比，标注同步变换"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    interp = cv.INTER_AREA if r < 1 else cv.INTER_LINEAR
    l, t = int(round((size - nw) / 2 - 0.1)), int(round((size - nh) / 2 - 0.1))
    pad = lambda a, v: cv.copyMakeBorder(a, t, size - nh - t, l, size - nw - l, cv.BORDER_CONSTANT, value=v)
    out = pad(cv.resize(img, (nw, nh), interpolation=interp) if (nw, nh) != (w, h) else img, (color,) * 3)
    new = []
    for c, g in items:
        if SEG:
            mm = (cv.resize(g * 255, (nw, nh), interpolation=interp) > 127).astype(np.uint8) if (nw, nh) != (w, h) else g
            mm = pad(mm, 0)
            if int(mm.sum()) >= max(1, C.min_inst_px // 4):
                new.append((c, mm))
        else:
            b = g * r + [l, t, l, t]
            new.append((c, np.clip(b, 0, size)))
    return out, new


def rand_matrix(cw, ch, out_size, degrees, translate, scale, shear, perspective):
    """随机 旋转+缩放+平移+错切(+透视) 的 3x3 矩阵，把 cw×ch 的输入放到 out_size×out_size 输出中央附近"""
    M0 = np.eye(3); M0[0, 2], M0[1, 2] = -cw / 2, -ch / 2
    P = np.eye(3); P[2, 0], P[2, 1] = random.uniform(-perspective, perspective), random.uniform(-perspective, perspective)
    R = np.eye(3); R[:2] = cv.getRotationMatrix2D((0, 0), random.uniform(-degrees, degrees), random.uniform(*scale))
    S = np.eye(3)
    S[0, 1] = np.tan(np.deg2rad(random.uniform(-shear, shear)))
    S[1, 0] = np.tan(np.deg2rad(random.uniform(-shear, shear)))
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_size
    return T @ S @ R @ P @ M0


def geo_matrix(cw, ch, size):
    """随机几何增强的矩阵：缩放以"整张图刚好放进 size×size"为基准（r0），再乘随机倍数。
    不乘 r0 的话，大图会被原尺寸裁成一小块（目标大量丢失/只剩局部）"""
    r0 = size / max(cw, ch)
    return rand_matrix(cw, ch, size, C.degrees, C.translate, (C.scale_lo * r0, C.scale_hi * r0), C.shear, C.perspective)


def warp(img, items, H, size, fill=114, min_vis=MIN_VIS):
    """同一个矩阵 H 变换图和标注。返回 (图, 新标注, 有效区域)。
    框：4 个角变换后取外接框，裁剪后可见面积 < min_vis 的丢掉。mask：变换后留在图内的面积 < 理论面积×min_vis 的丢掉"""
    persp = abs(H[2, 0]) > 0 or abs(H[2, 1]) > 0
    wp = (lambda a, fl, bv: cv.warpPerspective(a, H, (size, size), flags=fl, borderValue=bv)) if persp else \
         (lambda a, fl, bv: cv.warpAffine(a, H[:2], (size, size), flags=fl, borderValue=bv))
    out = wp(img, cv.INTER_LINEAR, (fill,) * 3)
    valid = wp(np.full(img.shape[:2], 255, np.uint8), cv.INTER_NEAREST, 0) > 0
    tf = lambda x, y: (lambda v: v[:2] / v[2])(H @ np.array([x, y, 1.0]))
    new = []
    for c, g in items:
        if SEG:
            ys, xs = np.nonzero(g)
            cx, cy = xs.mean(), ys.mean()
            p0, px, py = tf(cx, cy), tf(cx + 1, cy), tf(cx, cy + 1)
            det = abs((px - p0)[0] * (py - p0)[1] - (px - p0)[1] * (py - p0)[0])
            wm = (wp(g * 255, cv.INTER_LINEAR, 0) > 127).astype(np.uint8)
            n = int(wm.sum())
            if n >= max(1, C.min_inst_px // 4) and n >= min_vis * len(xs) * det:
                new.append((c, wm))
        else:
            x1, y1, x2, y2 = g
            pts = np.array([tf(x1, y1), tf(x2, y1), tf(x2, y2), tf(x1, y2)])
            b = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            a_t = (b[2] - b[0]) * (b[3] - b[1])
            cb = np.clip(b, 0, size)
            if cb[2] - cb[0] >= 2 and cb[3] - cb[1] >= 2 and (cb[2] - cb[0]) * (cb[3] - cb[1]) >= min_vis * a_t:
                new.append((c, cb))
    return out, new, valid


def rot_flip(img, items, lr=0.5, ud=0.0, rot90=False):
    """水平/垂直翻转，可选 90° 倍数旋转（正方形图，框/mask 无损）"""
    S = img.shape[1]
    if random.random() < lr:
        img = img[:, ::-1]
        items = [(c, (m[:, ::-1] if SEG else np.array([S - m[2], m[1], S - m[0], m[3]]))) for c, m in items]
    if random.random() < ud:
        img = img[::-1]
        items = [(c, (m[::-1] if SEG else np.array([m[0], S - m[3], m[2], S - m[1]]))) for c, m in items]
    if rot90 and img.shape[0] == img.shape[1]:
        for _ in range(random.randrange(4)):
            img = np.rot90(img)
            items = [(c, (np.rot90(m) if SEG else np.array([m[1], S - m[2], m[3], S - m[0]]))) for c, m in items]
    return np.ascontiguousarray(img), [(c, np.ascontiguousarray(m)) for c, m in items]


def tiles(img, items, size, overlap, keep_empty, min_vis):
    """滑窗切片（不缩放，小目标不会被缩没）。被切开的目标：可见部分 ≥ min_vis 才保留；无目标的切片只留 keep_empty 比例作负样本"""
    h, w = img.shape[:2]
    stride = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(w - size, 0) + 1, stride))
    ys = list(range(0, max(h - size, 0) + 1, stride))
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    area0 = [(int(g.sum()) if SEG else (g[2] - g[0]) * (g[3] - g[1])) for _, g in items]
    out = []
    for y0 in ys:
        for x0 in xs:
            tile = img[y0:y0 + size, x0:x0 + size]
            ph, pw = size - tile.shape[0], size - tile.shape[1]
            if ph or pw:
                tile = cv.copyMakeBorder(tile, 0, ph, 0, pw, cv.BORDER_CONSTANT, value=(114,) * 3)
            sub = []
            for (c, g), a0 in zip(items, area0):
                if SEG:
                    mm = g[y0:y0 + size, x0:x0 + size]
                    n = int(mm.sum())
                    if n >= max(1, C.min_inst_px // 4) and n >= min_vis * a0:
                        sub.append((c, cv.copyMakeBorder(mm, 0, ph, 0, pw, cv.BORDER_CONSTANT, value=0) if (ph or pw) else mm.copy()))
                else:
                    b = np.clip(g - [x0, y0, x0, y0], 0, size)
                    if b[2] - b[0] >= 2 and b[3] - b[1] >= 2 and (b[2] - b[0]) * (b[3] - b[1]) >= min_vis * a0:
                        sub.append((c, b))
            if not sub and random.random() > keep_empty:
                continue
            out.append((tile, sub))
    return out


def mosaic(samples, size):
    """4 张图拼 1 张再随机仿射成 size×size（不建大画布：每张图的"摆位 + 最终仿射"合成一个矩阵）"""
    xc, yc = [int(random.uniform(0.5 * size, 1.5 * size)) for _ in range(2)]
    M = rand_matrix(2 * size, 2 * size, size, 0, 0.1, (0.5, 1.5), 0, 0)
    out = np.full((size, size, 3), 114, np.uint8)
    items = []
    for i, (img, it) in enumerate(samples):
        h, w = img.shape[:2]
        r = size / max(h, w)
        x1 = xc - w * r if i in (0, 2) else xc
        y1 = yc - h * r if i in (0, 1) else yc
        A = np.array([[r, 0, x1], [0, r, y1], [0, 0, 1.0]])
        o, ni, valid = warp(img, it, M @ A, size)
        out[valid] = o[valid]
        items += ni
    return out, items


def rows_of(items, w, h):
    """标注 → YOLO 文本行。检测：c xc yc w h；分割：c x1 y1 x2 y2 …（外轮廓，approxPolyDP ≤ poly_eps 像素）"""
    rows = []
    for c, g in items:
        if SEG:
            cs, _ = cv.findContours(g, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            for ct in cs:
                if cv.contourArea(ct) + len(ct) / 2 < max(1, C.min_inst_px // 4):
                    continue
                ap = cv.approxPolyDP(ct, C.poly_eps, True).reshape(-1, 2)
                if len(ap) >= 3:
                    p = ((ap + 0.5) / [w, h]).clip(0, 1)
                    rows.append(f'{int(c)} ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in p))
        else:
            x1, y1, x2, y2 = np.clip(g, 0, [w, h, w, h])
            if x2 - x1 > 0 and y2 - y1 > 0:
                rows.append(f'{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
    return rows


def semantic_mask(items, w, h):
    sem = np.zeros((h, w), np.uint8)
    for c, m in items:
        sem[m > 0] = c + 1
    return sem


def crops_of(img, items):
    h, w = img.shape[:2]
    out = []
    for c, g in items:
        x, y, bw, bh = cv.boundingRect(g) if SEG else (g[0], g[1], g[2] - g[0], g[3] - g[1])
        x1, y1 = int(max(0, x - C.cls_margin * bw)), int(max(0, y - C.cls_margin * bh))
        x2, y2 = int(min(w, x + bw * (1 + C.cls_margin))), int(min(h, y + bh * (1 + C.cls_margin)))
        if x2 - x1 < C.cls_min_px or y2 - y1 < C.cls_min_px:
            continue
        crop, _ = lb(img[y1:y2, x1:x2], [], C.cls_size)
        out.append((c, crop))
    return out


# ===================== 逐像素处理（几何不变，标注原样保留）=====================
def to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def hist_eq(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def clahe_gray(g, clip=2.0, grid=8):
    return cv.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(g)


def clahe_color(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe_gray(lab[:, :, 0])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def scharr_mag(g):
    g = cv.GaussianBlur(g, (3, 3), 0).astype(np.float32)
    mag = cv.magnitude(cv.Scharr(g, cv.CV_32F, 1, 0), cv.Scharr(g, cv.CV_32F, 0, 1))
    return np.clip(mag / max(float(np.percentile(mag, 99.5)), 1e-6) * 255, 0, 255).astype(np.uint8)


def canny_auto(g):
    b = cv.GaussianBlur(g, (5, 5), 1.4)
    hi, _ = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return cv.Canny(b, 0.5 * hi, hi, L2gradient=True)


def laplacian_edge(g):
    lap = cv.Laplacian(cv.GaussianBlur(g, (0, 0), 1.5), cv.CV_32F, ksize=3)
    k = np.ones((3, 3), np.uint8)
    lo, hi = cv.erode(lap, k), cv.dilate(lap, k)
    sl = hi - lo
    return (((lap >= 0) & (lo < 0) & (sl > 0.25 * np.percentile(sl, 99))) * 255).astype(np.uint8)


def lbp_uniform(gray, radius=1, n_points=8, tol=4.0, sigma=0.7):
    g = cv.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    h, w = g.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    bits = np.empty((n_points, h, w), np.uint8)
    for p in range(n_points):
        a = 2 * np.pi * p / n_points
        nb = cv.remap(g, (xs + radius * np.cos(a)).astype(np.float32), (ys - radius * np.sin(a)).astype(np.float32), cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        bits[p] = (nb - g >= -tol)
    tr = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    return np.where(tr <= 2, bits.sum(axis=0), n_points + 1).astype(np.uint8)


def feat_stack(img):
    g = to_gray(img)
    return cv.merge((clahe_gray(g), scharr_mag(g), (lbp_uniform(g).astype(np.float32) / 9 * 255).astype(np.uint8)))


def gray_world(img):
    """灰度世界白平衡（各通道增益限制在 0.8~1.25）：缓和不同相机/光照造成的色偏。
    叶片图本来就以绿色为主，完全按灰度世界做会把绿色整个抹掉，所以限制了增益幅度。"""
    f = img.astype(np.float32)
    m = f.reshape(-1, 3).mean(0)
    return np.clip(f * np.clip(m.mean() / np.maximum(m, 1e-6), 0.8, 1.25), 0, 255).astype(np.uint8)


def feat_veg(img):
    """植被特征图（3 通道）：ExG 超绿指数 / 饱和度 / CLAHE 亮度。病斑 = 绿色里"不绿"的地方，ExG 会让它更突出（未在你的数据上验证过对精度的作用）"""
    f = img.astype(np.float32) + 1e-3
    s = f.sum(2, keepdims=True)
    b, g, r = [(f[:, :, i:i + 1] / s)[:, :, 0] for i in range(3)]
    exg = np.clip((2 * g - r - b + 1) / 3 * 255, 0, 255).astype(np.uint8)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return cv.merge((exg, hsv[:, :, 1], clahe_gray(cv.cvtColor(img, cv.COLOR_BGR2LAB)[:, :, 0])))


def _small(img, side=1024):
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    return (cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img), s


def dehaze(img, omega=0.85, t0=0.2):
    """暗通道先验去雾（He 2009 的简化版：透射率在缩小图上估计并高斯平滑，不用导向滤波）。
    只是一个增强变体，未在你的雾/烟图上验证过；烟太浓或天空大面积时可能偏色；没有明显雾（大气光亮度 < 0.6）的图原样返回。"""
    sm, s = _small(img, 800)
    f = sm.astype(np.float32) / 255
    k = max(3, int(15 * max(sm.shape[:2]) / 800) | 1)
    dark = cv.erode(f.min(2), np.ones((k, k), np.uint8))
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = f.reshape(-1, 3)[idx].mean(0)
    if A.mean() < 0.6:                          # 画面里没有"亮的大气光"区域 → 看不出有雾，原样返回（否则对清晰图会强行拉伸、严重偏色）
        return img
    t = 1 - omega * cv.erode((f / A).min(2), np.ones((k, k), np.uint8))
    t = cv.GaussianBlur(t, (0, 0), k / 3)
    t = cv.resize(np.clip(t, t0, 1), (img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
    J = (img.astype(np.float32) / 255 - A) / t[:, :, None] + A
    return np.clip(J * 255, 0, 255).astype(np.uint8)


def deglare(img):
    """抑制水面镜面反光：找"又亮又不饱和"的小片区域，在缩小图上 inpaint 再贴回。
    注意：白色漂浮物（泡沫塑料、白塑料袋）也满足"亮且不饱和"，会被一起抹掉——所以它只是可选的对照变体，默认不开。"""
    sm, s = _small(img, 1024)
    hsv = cv.cvtColor(sm, cv.COLOR_BGR2HSV)
    m = ((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 40)).astype(np.uint8)
    m = cv.dilate(m, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):                       # 太大的亮区（白色物体/天空）不动
        if st[i, cv.CC_STAT_AREA] > 0.01 * m.size:
            m[lab == i] = 0
    if not m.any():
        return img
    inp = cv.inpaint(sm, m, 3, cv.INPAINT_TELEA)
    if s < 1:
        inp = cv.resize(inp, (img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
        m = cv.resize(m, (img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)
    out = img.copy()
    out[m > 0] = inp[m > 0]
    return out


PIXEL_OPS = {      # 变体名 → (函数, 存储格式, letterbox 补边值)。先在原图上做、再 letterbox（补边处不会出现假边缘）；特征图用 png、补 0
    'hist_eq': (hist_eq, 'jpg', 114),
    'clahe': (clahe_color, 'jpg', 114),
    'median': (lambda im: cv.medianBlur(im, 5), 'jpg', 114),
    'gauss': (lambda im: cv.GaussianBlur(im, (3, 3), 1), 'jpg', 114),
    'edge_scharr': (lambda im: cv.cvtColor(scharr_mag(to_gray(im)), cv.COLOR_GRAY2BGR), 'png', 0),
    'edge_canny': (lambda im: cv.cvtColor(canny_auto(to_gray(im)), cv.COLOR_GRAY2BGR), 'png', 0),
    'edge_lap': (lambda im: cv.cvtColor(laplacian_edge(to_gray(im)), cv.COLOR_GRAY2BGR), 'png', 0),
    'feat_stack': (feat_stack, 'png', 0),
    'gray_world': (gray_world, 'jpg', 114),
    'feat_veg': (feat_veg, 'png', 0),
    'dehaze': (dehaze, 'jpg', 114),
    'deglare': (deglare, 'jpg', 114),
}


# ===================== 随机像素增强（仅 train；在 letterbox 之后做，标注不变）=====================
def hsv_jitter(img):
    h, s, v = C.hsv
    r = np.random.uniform(-1, 1, 3) * (h, s, v) + 1
    hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    return cv.cvtColor(cv.merge((cv.LUT(hue, ((x * r[0]) % 180).astype(np.uint8)), cv.LUT(sat, np.clip(x * r[1], 0, 255).astype(np.uint8)),
                                 cv.LUT(val, np.clip(x * r[2], 0, 255).astype(np.uint8)))), cv.COLOR_HSV2BGR)


def aug_color(img):
    img = hsv_jitter(img)
    g = random.uniform(*C.gamma)
    img = cv.LUT(img, (np.clip((np.arange(256) / 255.0) ** g, 0, 1) * 255).astype(np.uint8))
    return cv.convertScaleAbs(img, alpha=random.uniform(*C.contrast), beta=random.uniform(*C.bright))


def aug_degrade(img):
    for op in random.sample(['noise', 'motion', 'lowres', 'jpeg'], random.choice([1, 2])):
        if op == 'noise':
            img = np.clip(img.astype(np.float32) + np.random.normal(0, random.uniform(3, 12), img.shape), 0, 255).astype(np.uint8)
        elif op == 'motion':
            k = random.choice([5, 7, 9, 13])
            ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1
            ker = cv.warpAffine(ker, cv.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1), (k, k))
            img = cv.filter2D(img, -1, ker / max(float(ker.sum()), 1e-6))
        elif op == 'lowres':
            h, w = img.shape[:2]
            s = random.uniform(0.4, 0.7)
            img = cv.resize(cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA), (w, h), interpolation=cv.INTER_LINEAR)
        else:
            img = cv.imdecode(cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, random.randint(30, 70)])[1], cv.IMREAD_COLOR)
    return img


def _lowfreq(h, w, cells=6):
    n = np.random.rand(cells, max(2, int(cells * w / h))).astype(np.float32)
    return cv.GaussianBlur(cv.resize(n, (w, h), interpolation=cv.INTER_CUBIC), (0, 0), max(h, w) / 40)


def aug_glare(img):
    """水面反光：叠加 1~3 条拉长的柔和亮斑（屏幕混合），模拟太阳/天空在水面的镜面反射"""
    h, w = img.shape[:2]
    ov = np.zeros((h, w), np.float32)
    for _ in range(random.randint(1, 3)):
        c = (random.randint(0, w), random.randint(0, h))
        ax = (int(random.uniform(0.05, 0.3) * w), int(random.uniform(0.01, 0.05) * h))
        tmp = np.zeros((h, w), np.float32)
        cv.ellipse(tmp, c, ax, random.uniform(-20, 20), 0, 360, 1.0, -1)
        ov += cv.GaussianBlur(tmp, (0, 0), max(3, w / 60)) * random.uniform(0.5, 1.0)
    ov = np.clip(ov, 0, 1)[:, :, None] * 255
    return (255 - (255 - img.astype(np.float32)) * (255 - ov) / 255).clip(0, 255).astype(np.uint8)


def aug_haze(img):
    """雾 / 烟：I = J·t + A·(1-t)，t = exp(-β·深度)，深度 = 随机方向渐变 + 低频噪声（浓淡不均）"""
    h, w = img.shape[:2]
    th = random.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx / w - .5) * np.cos(th) + (yy / h - .5) * np.sin(th)) + .5
    d = np.clip(d + 0.6 * (_lowfreq(h, w) - .5), 0, 1.2)
    t = np.exp(-random.uniform(0.6, 2.5) * d)[:, :, None]
    A = np.array([random.uniform(190, 245)] * 3) * np.array([1.0, random.uniform(.96, 1.0), random.uniform(.92, 1.0)])
    return (img.astype(np.float32) * t + A * (1 - t)).clip(0, 255).astype(np.uint8)


def aug_shadow(img):
    """随机阴影（树影/云影/建筑影）：多边形区域变暗，边缘柔化"""
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    pts = (np.random.rand(random.randint(3, 6), 2) * [w, h]).astype(np.int32)
    cv.fillPoly(m, [cv.convexHull(pts)], 1.0)
    m = cv.GaussianBlur(m, (0, 0), max(3, w / 40))[:, :, None]
    return (img.astype(np.float32) * (1 - m * random.uniform(0.25, 0.6))).clip(0, 255).astype(np.uint8)


def aug_lowlight(img):
    """弱光 / 阴天 / 黄昏：压暗 + 提高对比度的暗部噪声"""
    f = (img.astype(np.float32) / 255) ** random.uniform(1.4, 2.4) * random.uniform(0.55, 0.9) * 255
    return np.clip(f + np.random.normal(0, random.uniform(2, 7), f.shape), 0, 255).astype(np.uint8)


RANDOM_PIX = {'color': aug_color, 'degrade': aug_degrade, 'glare': aug_glare, 'haze': aug_haze, 'shadow': aug_shadow, 'lowlight': aug_lowlight}
GEO_VARIANTS = ('tiles', 'aug_geo', 'aug_mosaic', 'aug_full')
TRAIN_ONLY = {'aug_geo', 'aug_mosaic', 'aug_full'} | {'aug_' + k for k in RANDOM_PIX}   # tiles 在所有集合里都生成（小目标场景下验证集也要切片才有意义）


# ===================== 6 生成：worker / 写盘 / yaml / 报告 / 自检 =====================
def stem_table(keys):
    """输出文件名：相对路径里的 '/' → '+'，Windows 不允许的字符 → '_'；仍重名的加 6 位哈希"""
    out, used = {}, {}
    for k in sorted(keys, key=natural_key):
        s = safe_name(key_to_stem(k))
        if re.search(r'__[kt]\d+$', s):
            s += '_s'                                  # 避免和增强图 __k0 / 切片 __t0 的命名撞车
        if s in used:
            s += '_' + hashlib.md5(k.encode('utf-8')).hexdigest()[:6]
        used[s] = k
        out[k] = s
    return out


def _save(out, variant, split, name, img, items, fmt='jpg'):
    h, w = img.shape[:2]
    d = os.path.join(out, variant)
    for sub in ('images', 'labels') + (('masks',) if SEG and C.export_masks else ()):
        os.makedirs(os.path.join(d, sub, split), exist_ok=True)
    imwrite_u(os.path.join(d, 'images', split, f'{name}.{fmt}'), img, [cv.IMWRITE_JPEG_QUALITY, 95] if fmt == 'jpg' else [])
    rows = rows_of(items, w, h)
    write_text(os.path.join(d, 'labels', split, f'{name}.txt'), '\n'.join(rows) + ('\n' if rows else ''))
    if SEG and C.export_masks:
        imwrite_u(os.path.join(d, 'masks', split, f'{name}.png'), semantic_mask(items, w, h))


def _load(sm):
    img = imread_u(sm['path'], cv.IMREAD_COLOR | (cv.IMREAD_IGNORE_ORIENTATION if sm['ignore_exif'] else 0))
    if img is None:
        return None, []
    proc, _ = _small(img, C.proc_max_side)
    return img, proc


def tiles_job(img, anns, size, rnd_keep):
    """切片。检测：框裁剪；分割：每个切片窗口里单独栅格化多边形（不在整张大图上建 mask，省内存）"""
    h, w = img.shape[:2]
    if not SEG:
        return tiles(img, to_items(anns, w, h), size, C.tile_overlap, C.tile_keep_empty, C.tile_min_vis)
    stride = max(1, int(size * (1 - C.tile_overlap)))
    xs = list(range(0, max(w - size, 0) + 1, stride))
    ys = list(range(0, max(h - size, 0) + 1, stride))
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    pp = [(a['cid'], np.asarray(a['geom']) * [w, h]) for a in anns]
    ar = [poly_area(p) for _, p in pp]
    out = []
    for y0 in ys:
        for x0 in xs:
            tile = img[y0:y0 + size, x0:x0 + size]
            ph, pw = size - tile.shape[0], size - tile.shape[1]
            if ph or pw:
                tile = cv.copyMakeBorder(tile, 0, ph, 0, pw, cv.BORDER_CONSTANT, value=(114,) * 3)
            sub = []
            for (c, p), a0 in zip(pp, ar):
                if p[:, 0].max() < x0 or p[:, 0].min() > x0 + size or p[:, 1].max() < y0 or p[:, 1].min() > y0 + size:
                    continue
                m = np.zeros((size, size), np.uint8)
                cv.fillPoly(m, [np.round((p - [x0, y0]) * 16).astype(np.int32)], 1, shift=4)
                if int(m.sum()) >= max(1, C.min_inst_px // 4) and m.sum() >= C.tile_min_vis * max(a0, 1):
                    sub.append((c, m))
            if not sub and random.random() > C.tile_keep_empty:
                continue
            out.append((tile, sub))
    return out


def process_one(t):
    """一张源图 → 所有变体的输出（在子进程里跑；随机种子 = 图的 key + 全局 seed，结果可复现、与进程数无关）"""
    fi = os.environ.get('PREP_FAULT_INJECT')            # 仅供测试：'片段:crash|raise'，key 含该片段的图会让 worker 崩溃/抛异常
    if fi and fi.rsplit(':', 1)[0] in t['key']:
        if fi.endswith(':crash'):
            os._exit(13)
        raise RuntimeError('injected fault')
    sd = zlib.crc32(t['key'].encode('utf-8')) ^ (C.seed * 2654435761 & 0xFFFFFFFF)
    random.seed(sd); np.random.seed(sd % (2 ** 32))
    cnt = collections.Counter()
    sm, split, name, out, V = t['sample'], t['split'], t['name'], t['out'], t['variants']
    img, proc = _load(sm)
    if img is None:
        raise InputError('图片读取失败（文件被移走/损坏/被占用？）')
    h, w = proc.shape[:2]
    items = to_items(sm['anns'], w, h)
    lbi, lbn = lb(proc, items, C.size)
    for it in lbn:
        g = it[1]
        a = float(g.sum()) if SEG else (g[2] - g[0]) * (g[3] - g[1])
        cnt[('inst', split)] += 1
        if a < 64:                                   # 缩放到 size×size 后面积 < 8×8 像素
            cnt[('tiny', split)] += 1

    def put(v, nm, im, its, fmt='jpg', sp=None):
        _save(out, v, sp or split, nm, im, its, fmt)
        cnt[(v, sp or split)] += 1
    if 'base' in V:
        put('base', name, lbi, lbn)
    for v, (fn, fmt, pad) in PIXEL_OPS.items():
        if v in V:
            pi, pn = lb(fn(proc), items, C.size, pad)
            put(v, name, pi, pn, fmt)
    if 'cls_crops' in V:
        for i, (c, crop) in enumerate(crops_of(proc, items)):
            d = os.path.join(out, 'cls_crops', split, safe_name(t['names'][c]))
            os.makedirs(d, exist_ok=True)
            imwrite_u(os.path.join(d, f'{name}_{i}.jpg'), crop, [cv.IMWRITE_JPEG_QUALITY, 95])
            cnt[('cls_crops', split)] += 1
    if 'tiles' in V and max(img.shape[:2]) >= C.tile_min_ratio * TILE:
        for i, (tl, tn) in enumerate(tiles_job(img, sm['anns'], TILE, None)):
            put('tiles', f'{name}__t{i}', tl, tn)
    if split != 'train':
        return cnt, None
    rp = [v for v in RANDOM_PIX if 'aug_' + v in V]
    for k in range(t['copies']):
        tag = f'{name}__k{k}'
        for v in rp:                                  # 像素类随机增强：标注不变
            put('aug_' + v, tag, RANDOM_PIX[v](lbi), lbn)

        def geo():
            a, an, _ = warp(proc, items, geo_matrix(w, h, C.size), C.size)
            return rot_flip(a, an, C.flip_lr, C.flip_ud, C.rot90)

        def mos():
            ps = t['partners'][k]
            if len(ps) < 3:
                return None
            others = []
            for p in ps:
                pi, pr = _load(p)
                if pi is None:
                    return None
                others.append((pr, to_items(p['anns'], pr.shape[1], pr.shape[0])))
            m, mn = mosaic([(proc, items)] + others, C.size)
            return rot_flip(m, mn, C.flip_lr, C.flip_ud, C.rot90)
        if 'aug_geo' in V:
            a, an = geo()
            if an or not items:
                put('aug_geo', tag, a, an)
        if 'aug_mosaic' in V:
            r = mos()
            if r and r[1]:
                put('aug_mosaic', tag, *r)
        if 'aug_full' in V:
            r = mos() if (t['partners'][k] and random.random() < 0.5) else None
            a, an = r if r else geo()
            if an or not items:
                for p in C.full_pix:
                    if p == 'color' or random.random() < 0.5:
                        a = RANDOM_PIX[p](a)
                put('aug_full', tag, a, an)
    return cnt, None


def write_yaml(root, variant, names, own_val):
    d = os.path.join(root, variant)
    # 不写 path（绝对路径）：路径相对于 yaml 所在文件夹，整个输出文件夹从 Mac 拷到 Windows 直接能训练
    val = 'images/val' if own_val else '../base/images/val'
    lines = ['train: images/train', f'val: {val}']
    if os.path.isdir(os.path.join(d, 'images', 'test')):
        lines.append('test: images/test')
    lines.append(f'nc: {len(names)}')
    lines.append('names: {' + ', '.join(f'{i}: {json.dumps(n, ensure_ascii=False)}' for i, n in enumerate(names)) + '}')
    write_text(os.path.join(d, 'data.yaml'), '\n'.join(lines) + '\n')


def self_check(root, variants, names, stem_key, assign, groups, samples, sizes_sample=200):
    """输出自检：① 每个标注文件格式/坐标/类别合法，图在 ② 图尺寸 ③ 泄漏：输出文件夹里每个源图所在集合 = 划分表，且同组、同 md5 的图没有跨集合"""
    bad, leak = [], 0
    seen = collections.defaultdict(set)
    imgs = []
    for v in variants:
        for lp in glob.glob(os.path.join(root, v, 'labels', '*', '*.txt')):
            sp = os.path.basename(os.path.dirname(lp))
            nm = os.path.splitext(os.path.basename(lp))[0]
            base = lp.replace(os.sep + 'labels' + os.sep, os.sep + 'images' + os.sep)[:-4]
            ip = next((base + e for e in ('.jpg', '.png') if os.path.exists(base + e)), None)
            if ip is None:
                bad.append((lp, '缺少对应的图'))
                continue
            imgs.append(ip)
            key = stem_key.get(re.sub(r'__[kt]\d+$', '', nm))
            if key is None:
                bad.append((lp, '文件名对不上任何源图'))
            else:
                seen[key].add(sp)
                if assign[key] != sp:
                    leak += 1
                    bad.append((lp, f'源图属于 {assign[key]}，却出现在 {sp}'))
            for line in open(lp, encoding='utf-8').read().splitlines():
                t = line.split()
                try:
                    v_ = [float(x) for x in t]
                except ValueError:
                    bad.append((lp, '非数字')); break
                n = len(v_) - 1
                ok = (n == 4) if not SEG else (n >= 6 and n % 2 == 0)
                if not ok or v_[0] != int(v_[0]) or not 0 <= v_[0] < len(names) or any(not (-1e-6 <= x <= 1 + 1e-6) for x in v_[1:]):
                    bad.append((lp, '列数/类别/坐标不合法：' + line[:40])); break
                if not SEG and (v_[3] <= 0 or v_[4] <= 0):
                    bad.append((lp, '宽高 ≤ 0')); break
    for ip in random.Random(0).sample(imgs, min(sizes_sample, len(imgs))):
        im = imread_u(ip)
        if im is None or im.shape[0] != im.shape[1]:
            bad.append((ip, '图损坏或不是正方形'))
    gsplit, msplit = {}, {}
    for k, sp in assign.items():
        if groups.get(k) and gsplit.setdefault(groups[k], sp) != sp:
            leak += 1; bad.append((k, '同一组跨集合'))
        if msplit.setdefault(samples[k]['md5'], sp) != sp:
            leak += 1; bad.append((k, '完全相同的图跨集合'))
    return bad, leak


def preview(root, variants, names, key0):
    cells = []
    for v in variants:
        ps = sorted(glob.glob(os.path.join(root, v, 'images', 'train', key0 + '*')))
        if not ps:
            continue
        im = imread_u(ps[0])
        h, w = im.shape[:2]
        lp = os.path.join(root, v, 'labels', 'train', os.path.splitext(os.path.basename(ps[0]))[0] + '.txt')
        over = im.copy()
        for line in (open(lp).read().strip().splitlines() if os.path.exists(lp) else []):
            t = [float(x) for x in line.split()]
            if SEG:
                pts = (np.array(t[1:]).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv.fillPoly(over, [pts], (0, 255, 0)); cv.polylines(im, [pts], True, (0, 255, 0), 2)
            else:
                xc, yc, bw, bh = t[1:]
                cv.rectangle(im, (int((xc - bw / 2) * w), int((yc - bh / 2) * h)), (int((xc + bw / 2) * w), int((yc + bh / 2) * h)), (0, 255, 0), 2)
        im = cv.addWeighted(over, 0.35, im, 0.65, 0) if SEG else im
        im = cv.resize(im, (320, int(320 * h / w)))
        cv_ = np.full((340, 320, 3), 30, np.uint8)
        cv_[20:20 + im.shape[0]] = im[:320]
        cv.putText(cv_, v, (4, 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cells.append(cv_)
    if not cells:
        return None
    while len(cells) % 4:
        cells.append(np.full_like(cells[0], 30))
    p = os.path.join(root, 'preview.jpg')
    imwrite_u(p, cv.vconcat([cv.hconcat(cells[i:i + 4]) for i in range(0, len(cells), 4)]), [cv.IMWRITE_JPEG_QUALITY, 90])
    return p


# ===================== 7 配置校验 / 环境检查 =====================
def validate_config():
    """一次性检查全部配置，把所有问题一起报出来（而不是改一个报一个）"""
    errs = []

    def need(cond, msg):
        if not cond:
            errs.append(msg)
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)
    isnum = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    need(C.task in ('detect', 'segment'), f"task 只能是 'detect' 或 'segment'，现在是 {C.task!r}")
    need(C.label_format in ('yolo', 'yolo_seg', 'coco', 'voc', 'mask', 'instmap'), f"label_format 不认识：{C.label_format!r}")
    need(C.strict in ('abort', 'skip', 'fix'), f"strict 只能是 'abort'/'skip'/'fix'，现在是 {C.strict!r}")
    need(C.missing_label in ('skip', 'background'), f"missing_label 只能是 'skip'/'background'，现在是 {C.missing_label!r}")
    need(C.group_mode in ('auto', 'folder', 'filename', 'visual', 'manifest', 'none'), f"group_mode 不认识：{C.group_mode!r}")
    need(C.exif in ('auto', 'apply', 'ignore'), f"exif 只能是 'auto'/'apply'/'ignore'，现在是 {C.exif!r}")
    need(isint(C.size) and C.size >= 32, f'size 必须是 ≥32 的整数，现在是 {C.size!r}')
    need(isint(TILE) and TILE >= 32, f'tile_size 必须是 ≥32 的整数，现在是 {C.tile_size!r}')
    need(isint(C.workers) and C.workers >= 1, f'workers 必须是 ≥1 的整数，现在是 {C.workers!r}')
    need(isint(C.copies) and C.copies >= 0, f'copies 必须是 ≥0 的整数，现在是 {C.copies!r}')
    need(isnum(C.oversample_max) and C.oversample_max >= 1, 'oversample_max 必须 ≥ 1')
    need(isinstance(C.ratios, (list, tuple)) and len(C.ratios) == 3 and all(isnum(r) and r >= 0 for r in C.ratios) and C.ratios[0] > 0,
         f'ratios 必须是 3 个非负数 (train, val, test)，且 train > 0，现在是 {C.ratios!r}')
    need(isnum(C.coord_tol) and 0 <= C.coord_tol <= 0.5, 'coord_tol 应在 0~0.5')
    need(isnum(C.max_bad_frac) and 0 <= C.max_bad_frac <= 1, 'max_bad_frac 应在 0~1')
    need(isnum(C.max_fail_frac) and 0 <= C.max_fail_frac <= 1, 'max_fail_frac 应在 0~1')
    need(isnum(C.scale_lo) and isnum(C.scale_hi) and 0 < C.scale_lo <= C.scale_hi, 'scale_lo/scale_hi 需要 0 < lo ≤ hi')
    need(isnum(C.degrees) and C.degrees >= 0, 'degrees 需要 ≥ 0')
    need(isinstance(C.hsv, (list, tuple)) and len(C.hsv) == 3, 'hsv 需要 3 个数 (色调, 饱和度, 亮度)')
    for nm in ('gamma', 'contrast', 'bright'):
        v = getattr(C, nm)
        need(isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1], f'{nm} 需要 (最小, 最大) 两个数且最小 ≤ 最大')
    valid = {'base', 'cls_crops', 'tiles'} | set(PIXEL_OPS) | TRAIN_ONLY
    bad = [v for v in C.variants if v not in valid]
    for v in bad:
        near = difflib.get_close_matches(v, sorted(valid), 1)
        errs.append(f'variants 里有不认识的 {v!r}' + (f'，是不是 {near[0]!r}？' if near else ''))
    need(len(C.variants) > 0, 'variants 不能为空')
    badp = [p for p in C.full_pix if p not in RANDOM_PIX]
    need(not badp, f'full_pix 里有不认识的 {badp}，可选：{sorted(RANDOM_PIX)}')
    need(os.path.isdir(C.img_dir), f'img_dir 不是文件夹：{C.img_dir}')
    if C.label_format == 'coco':
        need(os.path.isfile(C.ann_path), f'COCO 的 ann_path 应该是 json 文件：{C.ann_path}')
    else:
        need(os.path.isdir(C.ann_path), f'ann_path 不是文件夹：{C.ann_path}')
    if C.classes_file:
        need(os.path.isfile(C.classes_file), f'classes_file 不存在：{C.classes_file}')
    if C.group_manifest:
        need(os.path.isfile(C.group_manifest), f'group_manifest 不存在：{C.group_manifest}')
    if C.mask_values:
        need(isinstance(C.mask_values, dict) and all(isinstance(k, int) for k in C.mask_values), 'mask_values 应是 {像素值(整数): 类别名}')
    if errs:
        raise ConfigError(f'配置有 {len(errs)} 处问题：\n' + '\n'.join(f'   {i}) {e}' for i, e in enumerate(errs, 1)), '修改脚本顶部的 CFG（或命令行参数）后重试')


def check_environment():
    v = tuple(int(x) for x in re.findall(r'\d+', cv.__version__)[:2])
    if v < (4, 5):
        raise PrepError(f'OpenCV 版本过低（{cv.__version__}）', '运行 pip install -U opencv-python')
    for ext in ('.jpg', '.png'):
        ok, _ = cv.imencode(ext, np.zeros((8, 8, 3), np.uint8))
        if not ok:
            raise PrepError(f'这个 OpenCV 不支持 {ext} 编码', '重新安装：pip install --force-reinstall opencv-python')
    if Image is None:
        log('      提示：没装 Pillow，EXIF 方向/拍摄时间线索不可用（pip install pillow）')


def check_writable(folder):
    os.makedirs(folder, exist_ok=True)
    try:
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as e:
        raise OutputError(f'输出位置写不了：{folder}（{e}）', os_error_hint(e))


def estimate_disk(ctx):
    """粗估输出体积并和剩余空间比较（只警告：真正写满时 Failure(fatal) 会立刻终止）"""
    try:
        n = len(ctx.samples)
        per = C.size * C.size * 3 * 0.15
        pix = [v for v in C.variants if v in PIXEL_OPS or v == 'base']
        aug = [v for v in C.variants if v in TRAIN_ONLY]
        est = n * per * (len(pix) + 0.8 * C.copies * len(aug) * 1.3)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(ctx.out)) or '.').free
        if free < 1.5 * est:
            log(f'      ⚠ 粗估输出约 {est / 2**30:.1f}GB，剩余空间 {free / 2**30:.1f}GB：可能不够，先清理空间或减少 variants/copies')
    except Exception:                                                  # noqa
        pass


# ===================== 8 阶段函数（每个阶段只做一件事，出错有明确的异常类型）=====================
def new_context():
    return types.SimpleNamespace(t0=time.time(), out=C.out_dir, stage=None, rep=None, issues=Issues(), files={}, scans={}, samples={}, names=[], cmap=[],
                                 quarantined=[], groups={}, why={}, assign={}, split_rows=[], rf={}, stems={}, stem_key={}, replace_existing=False, failures=[])


def stage_prepare(ctx):
    """校验配置/环境，处理输出文件夹（正式运行先写到 out.partial，成功后再原子替换 out，失败不会破坏你已有的结果）"""
    validate_config()
    check_environment()
    out = ctx.out
    if C.check_only:
        ctx.rep = out + '_check'
        os.makedirs(ctx.rep, exist_ok=True)
        check_writable(ctx.rep)
    else:
        if os.path.exists(out) and os.listdir(out):
            if C.overwrite and os.path.exists(os.path.join(out, 'reports', 'run_info.json')):
                ctx.replace_existing = True
            else:
                raise OutputError(f'{out} 已存在且不是空的', '换个输出文件夹；或加 --overwrite（只会替换本脚本之前生成的结果，且新结果成功后才替换）。如果刚跑过 --check-only，它的报告在 ' + out + '_check，不影响')
        ctx.stage = out + '.partial'
        if os.path.exists(ctx.stage):
            log(f'      清理上次中断留下的 {ctx.stage}')
            shutil.rmtree(ctx.stage, ignore_errors=True)
        check_writable(os.path.dirname(os.path.abspath(out)) or '.')
        ctx.rep = os.path.join(ctx.stage, 'reports')
        os.makedirs(ctx.rep, exist_ok=True)
    open_log(os.path.join(ctx.rep, 'run.log'))
    log(f'数据预处理 {__version__} | task={C.task} format={C.label_format} | OpenCV {cv.__version__} | workers={C.workers}')


def stage_scan(ctx):
    ctx.files = discover_images(C.img_dir, ctx.issues)
    if not ctx.files:
        raise InputError('没找到图片：' + C.img_dir, f'支持的扩展名：{IMG_EXTS}；子文件夹会被递归读取')
    log(f'[1/6] 找到 {len(ctx.files)} 张图，扫描中…')
    items = list(ctx.files.items())
    for (k, _), r in zip(items, run_pool(scan_one, items, C.workers, '扫描')):
        ctx.scans[k] = dict(key=k, ok=False, err=r.msg) if isinstance(r, Failure) else r


def stage_labels(ctx):
    log('[2/6] 读取标注 + 严格检查…')
    ctx.samples, ctx.names, ctx.cmap = build_samples(ctx.files, ctx.scans, ctx.issues)


def stage_quarantine(ctx):
    """隔离有错误的图。strict=abort 有错就停；skip/fix 排除后继续；隔离比例过高说明配置错了，直接终止"""
    iss = ctx.issues
    bad = iss.bad_keys() - {'(全局)'}
    n_all = len(ctx.files)
    ctx.quarantined = sorted(bad & (set(ctx.files) | set(ctx.samples)), key=natural_key)
    unattached = [r for r in iss.rows if r[0] == 'ERROR' and r[2] not in ctx.files and r[2] != '(全局)']
    for k in bad:
        ctx.samples.pop(k, None)
    cl = collections.Counter(r[0] for r in iss.rows)
    log(f'      问题统计：ERROR {cl["ERROR"]} | FIXED(自动修复) {cl["FIXED"]} | DROP(丢弃的标注) {cl["DROP"]} | WARN {cl["WARN"]}')
    iss.save(os.path.join(ctx.rep, 'issues.csv'))
    save_csv(os.path.join(ctx.rep, 'class_map.csv'), ['编号', '类别名', '来源名称', '有效实例数'], ctx.cmap)
    write_text(os.path.join(ctx.rep, 'quarantine.txt'), '\n'.join(ctx.quarantined) + ('\n' if ctx.quarantined else ''))
    if ctx.quarantined or unattached:
        errs = collections.Counter(r[1] for r in iss.rows if r[0] == 'ERROR')
        log(f'      有 {len(ctx.quarantined)} 张图有 ERROR（占 {len(ctx.quarantined) / n_all:.0%}）；错误类型：{dict(errs)}；详见 {os.path.join(ctx.rep, "issues.csv")}')
        if C.strict == 'abort':
            raise InputError('strict="abort"：存在错误，已停止（没有生成任何数据）', '看 issues.csv 修好标注；或改成 strict="skip"（隔离坏图继续）/"fix"（再自动修复坐标超界）')
        if ctx.quarantined and len(ctx.quarantined) / n_all > C.max_bad_frac:
            raise InputError(f'被隔离的图占 {len(ctx.quarantined) / n_all:.0%}，超过 max_bad_frac={C.max_bad_frac:.0%}',
                             '这么高的比例多半是类别文件/标注格式/图片路径配错了，不是个别坏数据。先看 issues.csv；确认数据没问题再调大 max_bad_frac')
        log(f'      strict="{C.strict}"：这些图已排除，其余继续')
    if not ctx.samples:
        raise InputError('没有可用的样本', '看 issues.csv：所有图都被排除了（标注缺失？格式配错？）')
    log(f'      类别（{len(ctx.names)}）：' + str({i: n for i, n in enumerate(ctx.names)}))


def stage_split(ctx):
    log('[3/6] 分组 + 按类别均衡划分…')
    scans_ok = {k: ctx.scans[k] for k in ctx.samples if k in ctx.scans}
    ctx.groups, ctx.why = make_groups(ctx.samples, scans_ok, ctx.names, ctx.issues)
    ctx.assign = split_groups(ctx.samples, ctx.groups, ctx.names, ctx.issues)
    ctx.split_rows, n_img = split_report(ctx.samples, ctx.assign, ctx.names, ctx.issues)
    gsz = collections.Counter(ctx.groups.values())
    log(f'      {len(ctx.samples)} 张图 → {len(gsz)} 组（最大组 {max(gsz.values())} 张；合并线索：{dict(ctx.why)}）；划分 {dict(n_img)}')
    save_csv(os.path.join(ctx.rep, 'class_stats.csv'), ['类别', '集合', '实例数', '含该类的图数', '占该类总数'], ctx.split_rows)
    save_csv(os.path.join(ctx.rep, 'split_manifest.csv'), ['image', 'split', 'group', 'md5', '实例数'],
             [(k, ctx.assign[k], ctx.groups[k], ctx.samples[k]['md5'], len(ctx.samples[k]['anns'])) for k in sorted(ctx.samples, key=natural_key)])
    ctx.issues.save(os.path.join(ctx.rep, 'issues.csv'))
    log('      各类别实例数  类别 | train | val | test')
    for n in ctx.names:
        g = {sp: next((r[2] for r in ctx.split_rows if r[0] == n and r[1] == sp), 0) for sp in ('train', 'val', 'test')}
        log(f'        {n:<16}{g["train"]:>7}{g["val"]:>7}{g["test"]:>7}')
    for r in ctx.issues.rows:
        if r[0] == 'WARN' and r[1] in ('few_instances_in_split', 'giant_group', 'group_size_capped', 'too_few_groups', 'class_folders_ignored', 'class_without_instances'):
            log('      ⚠ ' + r[4])
    ctx.rf = repeat_factors(ctx.samples, ctx.assign, ctx.names)
    ctx.stems = stem_table(ctx.samples)
    ctx.stem_key = {v: k for k, v in ctx.stems.items()}


def build_tasks(ctx, V):
    tr_keys = sorted([k for k in ctx.samples if ctx.assign[k] == 'train'], key=natural_key)
    tasks = []
    for k in sorted(ctx.samples, key=natural_key):
        sp = ctx.assign[k]
        rnd = random.Random(zlib.crc32(k.encode('utf-8')) ^ C.seed)
        n_copy = 0
        if sp == 'train':
            x = C.copies * ctx.rf.get(k, 1.0)
            n_copy = int(x) + (1 if rnd.random() < x - int(x) else 0)
        partners = []
        for _ in range(n_copy):
            pool_ = [p for p in tr_keys if p != k]
            partners.append([ctx.samples[p] for p in rnd.sample(pool_, 3)] if len(pool_) >= 3 else [])
        tasks.append(dict(key=k, sample=ctx.samples[k], split=sp, name=ctx.stems[k], out=ctx.stage, variants=V, copies=n_copy, partners=partners, names=ctx.names))
    return tasks, tr_keys


OUT_RE = r'(__[kt]\d+)?\.(jpg|png|txt)$'


def purge_outputs(root, name):
    """某张图处理失败/崩溃时，把它已经写出的所有文件清掉：保证各变体里这张图要么都在、要么都不在"""
    pat = re.compile(re.escape(name) + OUT_RE)
    crop = re.compile(re.escape(name) + r'_\d+\.jpg$')
    n = 0
    for dp, _, fs in os.walk(root):
        parts = os.path.relpath(dp, root).split(os.sep)
        if parts[0] == 'reports':
            continue
        rx = crop if parts[0] == 'cls_crops' else pat
        for f in fs:
            if rx.match(f):
                try:
                    os.remove(os.path.join(dp, f))
                    n += 1
                except OSError:
                    pass
    return n


def stage_generate(ctx):
    log('[4/6] 生成各变体…')
    estimate_disk(ctx)
    V = set(C.variants)
    tasks, ctx.tr_keys = build_tasks(ctx, V)
    res = run_pool(process_one, tasks, C.workers, '生成')
    cnt, fails = collections.Counter(), []
    for t, r in zip(tasks, res):
        if isinstance(r, Failure):
            fails.append((t['key'], r))
            if r.fatal:
                raise OutputError(f'写文件时遇到不可恢复的错误：{r.msg}', r.tb or os_error_hint(OSError(errno.ENOSPC, '')))
        else:
            cnt.update(r[0])
    for k, f in fails:                                   # 失败的图：清理已写出的文件，并从划分里去掉
        purge_outputs(ctx.stage, ctx.stems[k])
        ctx.issues.add('ERROR', 'processing_failed', k, f.msg)
        ctx.failures.append((k, f))
    if fails:
        save_csv(os.path.join(ctx.rep, 'process_errors.csv'), ['image', 'error', 'crashed', 'detail'],
                 [(k, f.msg, f.crashed, (f.tb or '').replace('\n', ' | ')[:600]) for k, f in fails])
        log(f'      ⚠ {len(fails)} 张图处理失败（已从所有变体里清理），详见 process_errors.csv')
        for k, f in fails[:3]:
            log(f'        {k}：{f.msg}')
        if len(fails) / len(tasks) > C.max_fail_frac:
            raise PrepError(f'处理失败的图占 {len(fails) / len(tasks):.0%}，超过 max_fail_frac={C.max_fail_frac:.0%}，已终止',
                            f'未完成的结果留在 {ctx.stage} 供排查；先看 process_errors.csv，常见原因：内存不足（减小 workers/proc_max_side）、文件被占用、磁盘问题')
        for k, _ in fails:
            ctx.samples.pop(k, None)
            ctx.assign.pop(k, None)
        ctx.quarantined += [k for k, _ in fails]
    ctx.cnt, ctx.V = cnt, V


def stage_finalize(ctx):
    log('[5/6] 写 data.yaml / 报告…')
    V, out, stage = ctx.V, ctx.out, ctx.stage
    vlist = [v for v in ['base'] + list(PIXEL_OPS) + ['tiles'] + sorted(TRAIN_ONLY) if v in V and os.path.isdir(os.path.join(stage, v))]
    for v in vlist:
        write_yaml(stage, v, ctx.names, own_val=(v not in TRAIN_ONLY) or 'base' not in V)
    write_text(os.path.join(stage, 'classes.txt'), '\n'.join(ctx.names) + '\n')
    inst = sum(c for (a, s), c in ctx.cnt.items() if a == 'inst')
    tiny = sum(c for (a, s), c in ctx.cnt.items() if a == 'tiny')
    if inst and tiny / inst > 0.2:
        log(f'      ⚠ 有 {tiny / inst:.0%} 的目标缩放到 {C.size}px 后小于 8×8：小目标很多，建议在 variants 里打开 tiles，或把 size 调大')
    log(f'\n{"变体":<14}{"train":>8}{"val":>8}{"test":>8}')
    for v in vlist + (['cls_crops'] if 'cls_crops' in V else []):
        log(f'{v:<14}{ctx.cnt[(v, "train")]:>8}{ctx.cnt[(v, "val")]:>8}{ctx.cnt[(v, "test")]:>8}')
    log('[6/6] 输出自检（格式 / 泄漏）…')
    try:
        bad_o, leak = self_check(stage, vlist, ctx.names, ctx.stem_key, ctx.assign, ctx.groups, ctx.samples)
    except Exception as e:                                          # noqa  自检本身出错不应让整个流程失败
        bad_o, leak = [('(自检程序)', f'{type(e).__name__}: {e}')], 0
    save_csv(os.path.join(ctx.rep, 'selfcheck.csv'), ['file', 'problem'], bad_o)
    log('      自检：' + ('全部通过，没有泄漏' if not bad_o else f'发现 {len(bad_o)} 个问题（泄漏 {leak}），见 selfcheck.csv'))
    info = dict(version=__version__, config={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(C).items()}, classes=ctx.names,
                n_images=len(ctx.samples), n_quarantined=len(ctx.quarantined), n_failed=len(ctx.failures), opencv=cv.__version__, numpy=np.__version__,
                python=sys.version.split()[0], seconds=round(time.time() - ctx.t0, 1), counts={f'{a}/{s}': c for (a, s), c in ctx.cnt.items()})
    write_text(os.path.join(ctx.rep, 'run_info.json'), json.dumps(info, ensure_ascii=False, indent=1, default=str))
    ctx.issues.save(os.path.join(ctx.rep, 'issues.csv'))
    prev = None
    try:
        if ctx.tr_keys:
            prev = preview(stage, vlist, ctx.names, ctx.stems[ctx.tr_keys[0]])
    except Exception as e:                                          # noqa
        log(f'      预览图生成失败（不影响数据）：{e}')
    log(f'完成，用时 {time.time() - ctx.t0:.1f}s。')
    close_log()                                                     # Windows 上文件夹里有打开的文件不能改名
    try:
        if ctx.replace_existing:
            shutil.rmtree(out)
        os.replace(stage, out)
    except OSError as e:
        raise OutputError(f'把 {stage} 改名为 {out} 失败：{e}', (os_error_hint(e) or '关闭占用该文件夹的程序（资源管理器/杀毒/预览）后，手动把 .partial 改名即可') + f'；数据已完整生成在 {stage}')
    print(f'输出：{out}  报告：{os.path.join(out, "reports")}')
    if prev:
        print('预览图（绿色是标注，应当贴着目标）：', os.path.join(out, 'preview.jpg'))
    if 'base' in V:
        print('训练示例：yolo ' + ('segment' if SEG else 'detect') + f' train data={os.path.join(out, "base", "data.yaml")} model=yolo11n{"-seg" if SEG else ""}.pt imgsz={C.size}')


def write_crash_log(path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'数据预处理 {__version__} 崩溃日志\n时间 {time.strftime("%Y-%m-%d %H:%M:%S")}\nPython {sys.version}\nOpenCV {cv.__version__} NumPy {np.__version__} Pillow {"有" if Image else "无"}\n'
                    f'平台 {sys.platform}\n命令行 {sys.argv}\n\n配置:\n{json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(C).items()}, ensure_ascii=False, indent=1, default=str)}\n\n'
                    f'最近日志:\n' + '\n'.join(LOGBUF) + '\n\n' + traceback.format_exc())
        return path
    except Exception:                                               # noqa
        return None


def run():
    ctx = new_context()
    stage_prepare(ctx)
    stage_scan(ctx)
    stage_labels(ctx)
    stage_quarantine(ctx)
    stage_split(ctx)
    if C.check_only:
        log(f'[--check-only] 只做了检查，没有生成数据集；报告在 {ctx.rep}。用时 {time.time() - ctx.t0:.1f}s')
        return ctx
    stage_generate(ctx)
    stage_finalize(ctx)
    return ctx


def main():
    """入口：把"预期内的失败"变成清晰的中文提示 + 退出码 2；中断 130；其余未预料的异常写崩溃日志（退出码 1）"""
    random.seed(C.seed)
    np.random.seed(C.seed % (2 ** 32))
    try:
        run()
    except KeyboardInterrupt:
        log('\n已被用户中断。' + (f'未完成的结果留在 {C.out_dir}.partial（已有的 {C.out_dir} 没被改动）。' if not C.check_only else ''))
        close_log()
        sys.exit(130)
    except PrepError as e:
        msg = f'\n✖ {e}' + (f'\n  建议：{e.hint}' if e.hint else '')
        log(msg)
        close_log()
        sys.exit(e.exit_code)
    except SystemExit:
        raise
    except Exception as e:                                          # noqa
        p = write_crash_log(C.out_dir + '_crash.log')
        log(f'\n✖ 程序出现了未预料的错误：{type(e).__name__}: {e}')
        log(f'  详细信息已保存到 {p}，把这个文件内容发给我可以快速定位。')
        close_log()
        sys.exit(1)
    finally:
        close_log()



if __name__ == '__main__':
    main()

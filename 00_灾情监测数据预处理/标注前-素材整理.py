"""
标注前：原始素材整理（图片 + 视频 → 干净、统一、便于标注的图片集）
做的事（每一步都可在 CFG 里调）：
  1 视频抽帧：按间隔（interval）或画面变化（scene）抽，记录来源视频/帧号；每段视频一个文件夹（后面划分数据集时同一个视频不会被拆开）
  2 体检：损坏 / 太小 / 纯色 / 模糊 / 过暗 / 过曝；EXIF 旋转直接"烘焙"进像素（标注工具和训练框架对 EXIF 的处理不一致，是坐标错位的常见原因）
  3 去重：完全相同(md5) + 近似重复(视频里一动不动的几十帧只留一张)
  4 可选：大图缩小（max_side）；统一命名；Windows 不能用的字符自动替换
  5 标注优先级：先标"最有代表性"的图（最远点采样），可按批次分发给不同的人
  6 输出：images/（整理好的图）、group_manifest.csv（标注后脚本的 --manifest 直接用）、quality.csv（每张图的去留和原因）、rename_map 在 quality.csv 里
只需要 opencv-python + numpy（装了 pillow 会更准地读 EXIF 和尺寸）。
用法（python 标注前-素材整理.py --help 查看全部参数）：python 标注前-素材整理.py --in 原始素材文件夹 --out pre_out [--mode scene] [--every 1.0] [--batch-size 200] [--workers 4]
下一步：把 pre_out/images 导入 CVAT 等工具标注；标完后用"标注后-数据集生成"脚本，并加 --manifest pre_out/group_manifest.csv
"""

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
    in_dir='raw', out_dir='pre_out',
    video_mode='interval',          # 'interval' 每 every_sec 秒一帧 | 'scene' 画面变化够大才抽（静止镜头抽得少，运动镜头抽得多）
    every_sec=1.0, scene_thr=0.80,  # scene：与上一张已存帧的直方图相关系数 < scene_thr 才存
    min_gap_sec=0.3, max_gap_sec=5.0, max_frames_per_video=0,   # 0 = 不限；到上限后这段视频后面的部分不再抽
    min_side=224,                   # 短边小于它的图不要
    max_side=0,                     # >0：长边超过它的图缩小（标注更流畅）。注意：小目标（漂浮物/病斑）缩小会丢细节，默认不缩
    blur_min=15.0,                  # 拉普拉斯方差（在长边 512 的缩略图上算）低于它算模糊
    dark_mean=30, bright_mean=225, flat_std=6,
    reject=['corrupt', 'too_small', 'flat', 'exact_duplicate', 'near_duplicate'],   # 命中这些就剔除；'blur' 'dark' 'bright' 默认只标记不剔除，可自己加进来
    dup_hash_thr=0.08, dup_corr_thr=0.985, dup_color=12,       # 近似重复：哈希差 ≤8% 且 缩略图相关 ≥0.985 且 平均颜色接近（三个条件同时满足）
    batch_size=0, batch_mode='list',  # >0：按优先级分批；'list' 只写清单 | 'copy' 复制到 batches/batch_001/…
    priority_n=300,                 # 最多对前多少张做"代表性排序"
    max_fail_frac=0.2,              # 处理失败（读不出/写不了）的素材超过这个比例就终止
    jpg_quality=95, workers=1, seed=0, overwrite=False,
)


def _cli(cfg):
    ap = argparse.ArgumentParser(add_help=False)
    spec = (('in', 'in_dir', str), ('out', 'out_dir', str), ('mode', 'video_mode', str), ('every', 'every_sec', float),
            ('batch-size', 'batch_size', int), ('workers', 'workers', int), ('max-side', 'max_side', int))
    for k, _, t in spec:
        ap.add_argument('--' + k, type=t)
    for f in ('--overwrite', '--help', '--version'):
        ap.add_argument(f, action='store_true')
    ap.add_argument('-h', action='store_true', dest='help')
    a, _ = ap.parse_known_args()
    if a.version:
        print('标注前-素材整理 ' + __version__)
        raise SystemExit(0)
    if a.help:
        print(globals().get('__doc__') or '')
        print('命令行参数：--in 原始素材文件夹 --out 输出 --mode interval|scene --every 秒 --batch-size N --max-side 像素 --workers N --overwrite --version')
        raise SystemExit(0)
    cfg = {**DEFAULTS, **cfg}
    for k, ck, _ in spec:
        v = getattr(a, k.replace('-', '_'))
        if v is not None:
            cfg[ck] = v
    if a.overwrite:
        cfg['overwrite'] = True
    return cfg


C = types.SimpleNamespace(**_cli(CFG))
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
VID_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm')
KNOWN_REASONS = ('corrupt', 'too_small', 'flat', 'blur', 'dark', 'bright', 'exact_duplicate', 'near_duplicate')


# ---------------- 配置校验 ----------------
def validate_config():
    errs = []
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if not os.path.isdir(C.in_dir):
        errs.append(f'输入文件夹不存在：{C.in_dir}')
    if C.video_mode not in ('interval', 'scene'):
        errs.append(f"video_mode 只能是 'interval' 或 'scene'，现在是 {C.video_mode!r}")
    if C.batch_mode not in ('list', 'copy'):
        errs.append(f"batch_mode 只能是 'list' 或 'copy'")
    for nm in ('every_sec', 'min_gap_sec', 'max_gap_sec', 'min_side', 'blur_min', 'jpg_quality', 'workers'):
        if not num(getattr(C, nm)) or getattr(C, nm) < 0:
            errs.append(f'{nm} 需要是非负数，现在是 {getattr(C, nm)!r}')
    if C.workers < 1:
        errs.append('workers 需要 ≥ 1')
    if not (1 <= C.jpg_quality <= 100):
        errs.append('jpg_quality 需要在 1~100')
    if C.every_sec <= 0:
        errs.append('every_sec 需要 > 0')
    bad = [r for r in C.reject if r not in KNOWN_REASONS]
    if bad:
        errs.append(f'reject 里有不认识的 {bad}，可选：{list(KNOWN_REASONS)}')
    if errs:
        raise ConfigError(f'配置有 {len(errs)} 处问题：\n' + '\n'.join(f'   {i}) {e}' for i, e in enumerate(errs, 1)), '修改脚本顶部的 CFG（或命令行参数）后重试')


# ---------------- 视频抽帧 ----------------
def extract_video(a):
    """抽一段视频（在子进程里跑，单段视频出错只影响它自己）。返回 (组名, [(文件, 帧号, 秒)], 备注)"""
    path, stem, dst = a
    cap = cv.VideoCapture(path)
    tmp = None
    if not cap.isOpened():                           # Windows 上 VideoCapture 打不开含中文的路径：复制成英文名再试
        tmp = os.path.join(tempfile.gettempdir(), f'pre_{zlib.crc32(path.encode("utf-8")):08x}{os.path.splitext(path)[1]}')
        shutil.copy2(path, tmp)
        cap = cv.VideoCapture(tmp)
    try:
        if not cap.isOpened():
            raise InputError('无法打开视频', '编码不支持：先用 ffmpeg 转成 mp4/H.264（ffmpeg -i 原.mov -c:v libx264 新.mp4）')
        fps = cap.get(cv.CAP_PROP_FPS)
        note = ''
        if not fps or fps <= 0 or fps > 240:
            fps, note = 25.0, '读不到帧率，按 25fps 估计'
        os.makedirs(dst, exist_ok=True)
        out, i, last_t, last_h, bad_reads = [], -1, -1e9, None, 0
        probe = max(1, int(round(0.2 * fps)))
        while True:
            if not cap.grab():
                break
            i += 1
            t = i / fps
            if t - last_t < C.min_gap_sec:
                continue
            if C.video_mode == 'interval':
                if t - last_t < C.every_sec - 0.5 / fps:
                    continue
                ok, fr = cap.retrieve()
            else:
                if last_h is not None and i % probe:
                    continue
                ok, fr = cap.retrieve()
                if ok and fr is not None:
                    sm = cv.resize(fr, (160, max(1, int(160 * fr.shape[0] / fr.shape[1]))))
                    h = cv.calcHist([cv.cvtColor(sm, cv.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
                    cv.normalize(h, h)
                    if last_h is not None and t - last_t < C.max_gap_sec and cv.compareHist(last_h, h, cv.HISTCMP_CORREL) >= C.scene_thr:
                        continue
                    last_h = h
            if not ok or fr is None:
                bad_reads += 1                       # 个别坏帧跳过，不让整段视频失败
                continue
            fn = os.path.join(dst, f'{stem}_f{i:06d}.jpg')
            imwrite_u(fn, fr, [cv.IMWRITE_JPEG_QUALITY, C.jpg_quality])
            out.append((fn, i, round(t, 2)))
            last_t = t
            if C.max_frames_per_video and len(out) >= C.max_frames_per_video:
                break
        if bad_reads:
            note = (note + '；' if note else '') + f'{bad_reads} 帧解码失败已跳过'
        return stem, out, note
    finally:
        cap.release()
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------- 体检 ----------------
def scan(path):
    """读一张图的体检指标（尺寸/EXIF/清晰度/亮度/哈希/平均色）。失败返回 ok=False + 原因，不抛异常"""
    r = dict(path=path, ok=False)
    try:
        data = read_bytes(path)
        if data.size == 0:
            r['err'] = '空文件'
            return r
        r['md5'] = hashlib.md5(data).hexdigest()
        ori, raw = 1, None
        if Image is not None:
            try:
                with Image.open(path) as im:
                    raw, ori = im.size, int(im.getexif().get(274, 1) or 1)
            except Exception:                        # noqa
                raw = None
        if raw is None:
            full = cv.imdecode(data, cv.IMREAD_COLOR)
            if full is None:
                r['err'] = '无法解码（损坏，或 HEIC 等 OpenCV 不支持的格式）'
                return r
            eff, small = (full.shape[1], full.shape[0]), full
        else:
            eff = (raw[1], raw[0]) if ori in (5, 6, 7, 8) else raw
            k = max(1, min(8, int(max(raw) // 512)))
            fl = {1: cv.IMREAD_COLOR, 2: cv.IMREAD_REDUCED_COLOR_2, 4: cv.IMREAD_REDUCED_COLOR_4, 8: cv.IMREAD_REDUCED_COLOR_8}[1 if k < 2 else 2 if k < 4 else 4 if k < 8 else 8]
            small = cv.imdecode(data, fl)
            if small is None:
                r['err'] = '无法解码（损坏）'
                return r
        s = 512 / max(small.shape[:2])
        sm = cv.resize(small, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else small
        g = cv.cvtColor(sm, cv.COLOR_BGR2GRAY)
        r.update(ori=ori, eff=eff, blur=float(cv.Laplacian(g, cv.CV_64F).var()), mean=float(g.mean()), std=float(g.std()))
        t = cv.resize(g, (16, 16), interpolation=cv.INTER_AREA).astype(np.float32).ravel()
        r['th'] = ((t - t.mean()) / (t.std() + 1e-6)).astype(np.float32)
        d = cv.resize(g, (17, 16), interpolation=cv.INTER_AREA)
        r['dh'] = np.packbits((d[:, 1:] > d[:, :-1]).ravel())
        r['cm'] = cv.resize(sm, (8, 8), interpolation=cv.INTER_AREA).reshape(-1, 3).mean(0).astype(np.float32)
        r['ok'] = True
    except MemoryError:
        r['err'] = '内存不足（图片过大）'
    except Exception as e:                           # noqa
        r['err'] = f'{type(e).__name__}: {e}'
    return r


def dup_edges(DH, TH, CM):
    """严格的近似重复（三个条件同时满足）。返回 (后者下标, 前者下标)"""
    B = np.unpackbits(DH, axis=1).astype(np.float32)
    n = len(B)
    ei, ej = [], []
    for i0 in range(0, n, 512):
        i1 = min(n, i0 + 512)
        ham = 256 - (B[i0:i1] @ B[:i1].T + (1 - B[i0:i1]) @ (1 - B[:i1]).T)
        cor = TH[i0:i1] @ TH[:i1].T / TH.shape[1]
        cd = np.abs(CM[i0:i1, None, :] - CM[None, :i1, :]).max(2) <= C.dup_color
        m = (ham <= C.dup_hash_thr * 256) & (cor >= C.dup_corr_thr) & cd
        ii, jj = np.nonzero(m)
        sel = jj < ii + i0
        ei.append(ii[sel] + i0)
        ej.append(jj[sel])
    return np.concatenate(ei), np.concatenate(ej)


def put(a):
    """写出一张整理好的图：不需要处理的原样复制（不二次压缩）；有 EXIF 旋转/要缩小/格式特殊的才重新编码"""
    src, dst, ori, eff = a
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ext = os.path.splitext(dst)[1].lower()
    need = ori != 1 or (C.max_side and max(eff) > C.max_side) or ext not in ('.jpg', '.png')
    if not need:
        retry_io(lambda: shutil.copy2(src, dst))
        return True
    im = imread_u(src)                               # imdecode 会应用 EXIF 方向：旋转烘焙进像素，写出后不再带 EXIF
    if im is None:
        raise InputError('重新编码时读取失败')
    if C.max_side and max(im.shape[:2]) > C.max_side:
        s = C.max_side / max(im.shape[:2])
        im = cv.resize(im, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    imwrite_u(dst, im, [cv.IMWRITE_JPEG_QUALITY, C.jpg_quality] if ext == '.jpg' else [])
    return True


# ---------------- 阶段函数 ----------------
def new_context():
    return types.SimpleNamespace(t0=time.time(), out=C.out_dir, stage=None, items=[], keep=[], replace_existing=False, failed=[])


def stage_prepare(ctx):
    validate_config()
    out = ctx.out
    if os.path.exists(out) and os.listdir(out):
        if C.overwrite and os.path.exists(os.path.join(out, 'quality.csv')):
            ctx.replace_existing = True
        else:
            raise OutputError(f'{out} 已存在且不是空的', '换个输出文件夹，或加 --overwrite（只替换本脚本之前生成的结果，新结果成功后才替换）')
    ctx.stage = out + '.partial'
    if os.path.exists(ctx.stage):
        shutil.rmtree(ctx.stage, ignore_errors=True)
    parent = os.path.dirname(os.path.abspath(out)) or '.'
    try:
        os.makedirs(ctx.stage, exist_ok=True)
        with tempfile.TemporaryFile(dir=ctx.stage):
            pass
    except OSError as e:
        raise OutputError(f'输出位置写不了：{parent}（{e}）', os_error_hint(e))
    open_log(os.path.join(ctx.stage, 'run.log'))
    log(f'标注前-素材整理 {__version__} | OpenCV {cv.__version__} | workers={C.workers}')


def stage_collect(ctx):
    imgs, vids = [], []
    for root, _, names in os.walk(C.in_dir, onerror=lambda e: log(f'      ⚠ 文件夹读取失败：{e}')):
        for n in sorted(names, key=natural_key):
            if n.startswith('.') or n.startswith('._'):
                continue
            p = os.path.join(root, n)
            rel = os.path.relpath(p, C.in_dir).replace(os.sep, '/')
            if n.lower().endswith(IMG_EXTS):
                imgs.append((p, rel))
            elif n.lower().endswith(VID_EXTS):
                vids.append((p, rel))
    log(f'[1/5] 找到图片 {len(imgs)} 张，视频 {len(vids)} 段')
    for p, rel in imgs:
        d = rel.rsplit('/', 1)[0] if '/' in rel else ''
        ctx.items.append(dict(src=p, rel=rel, group=safe_name(d.replace('/', '+')), meta=''))
    return vids


def stage_extract(ctx, vids):
    frames_dir = os.path.join(ctx.stage, '_frames')
    jobs = []
    for p, rel in vids:
        stem = safe_name(os.path.splitext(rel)[0].replace('/', '+'))
        jobs.append((p, stem, os.path.join(frames_dir, stem)))
    for (p, rel), res in zip(vids, run_pool(extract_video, jobs, C.workers, '抽帧')):
        if isinstance(res, Failure):
            log(f'      ⚠ {rel}：{res.msg}' + (f'（{res.tb}）' if res.tb and len(res.tb) < 160 else ''))
            ctx.failed.append((rel, res.msg))
            continue
        stem, fr, note = res
        if note:
            log(f'      ⚠ {rel}：{note}')
        for fn, i, t in fr:
            ctx.items.append(dict(src=fn, rel=f'{rel}#frame{i}', group=stem, meta=f'video={rel};frame={i};t={t}s'))
        log(f'      {rel}：抽出 {len(fr)} 帧')
    return frames_dir


def stage_scan_filter(ctx):
    if not ctx.items:
        raise InputError('没有可用的素材', f'输入文件夹里没找到图片或视频：{C.in_dir}（支持 {IMG_EXTS + VID_EXTS}）')
    ctx.items.sort(key=lambda d: (natural_key(d['group']), natural_key(d['rel'])))
    log(f'[2/5] 体检 {len(ctx.items)} 张…')
    for d, r in zip(ctx.items, run_pool(scan, [d['src'] for d in ctx.items], C.workers, '体检')):
        d['r'] = dict(path=d['src'], ok=False, err=r.msg) if isinstance(r, Failure) else r
        d['issues'] = []
    seen = {}
    for d in ctx.items:
        r, iss = d['r'], d['issues']
        if not r['ok']:
            iss.append('corrupt'); d['why'] = r.get('err', '')
            continue
        if min(r['eff']) < C.min_side:
            iss.append('too_small')
        if r['std'] < C.flat_std:
            iss.append('flat')
        if r['blur'] < C.blur_min:
            iss.append('blur')
        if r['mean'] < C.dark_mean:
            iss.append('dark')
        if r['mean'] > C.bright_mean:
            iss.append('bright')
        if r['md5'] in seen:
            iss.append('exact_duplicate'); d['why'] = '与 ' + seen[r['md5']] + ' 完全相同'
        else:
            seen[r['md5']] = d['rel']
    rejected = lambda d: any(x in C.reject for x in d['issues'])
    elig = [d for d in ctx.items if d['r']['ok'] and not rejected(d)]
    if len(elig) > 1:
        DH, TH, CM = (np.stack([d['r'][k] for d in elig]) for k in ('dh', 'th', 'cm'))
        ei, ej = dup_edges(DH, TH, CM)
        nb = collections.defaultdict(list)
        for a, b in zip(ei, ej):
            nb[a].append(b)
            nb[b].append(a)
        kept = np.zeros(len(elig), bool)
        for i in sorted(range(len(elig)), key=lambda i: -elig[i]['r']['blur']):       # 近似重复里优先留最清晰的那张
            hit = next((j for j in nb.get(i, []) if kept[j]), None)
            if hit is None:
                kept[i] = True
            else:
                elig[i]['issues'].append('near_duplicate')
                elig[i]['why'] = '与 ' + elig[hit]['rel'] + ' 近似重复（保留了更清晰的那张）'
    ctx.keep = [d for d in ctx.items if d['r']['ok'] and not rejected(d)]
    ctx.cnt = collections.Counter(x for d in ctx.items for x in d['issues'])
    log(f'[3/5] 保留 {len(ctx.keep)} / {len(ctx.items)}')
    log('      命中情况（reject 里的会被剔除，其余只是标记）：' + str(dict(ctx.cnt)))
    if not ctx.keep:
        raise InputError('所有素材都被剔除了', '看 quality.csv 的原因；阈值（min_side / blur_min / dup_*）可能太严')


def stage_write_images(ctx):
    per = collections.defaultdict(int)
    for d in ctx.keep:
        g = d['group']
        per[g] += 1
        ext = os.path.splitext(d['src'])[1].lower()
        ext = '.jpg' if ext == '.jpeg' else ext
        ext = ext if ext in ('.jpg', '.png') else '.jpg'
        base = f'{g}_{per[g]:05d}' if g else f'img_{per[g]:05d}'
        d['new'] = (f'{g}/{base}' if g else base) + ext
    log('[4/5] 写出整理后的图片…')
    res = run_pool(put, [(d['src'], os.path.join(ctx.stage, 'images', d['new']), d['r']['ori'], d['r']['eff']) for d in ctx.keep], C.workers, '写出')
    ok = []
    for d, r in zip(ctx.keep, res):
        if isinstance(r, Failure):
            if r.fatal:
                raise OutputError(f'写文件时遇到不可恢复的错误：{r.msg}', r.tb)
            d['issues'].append('write_failed')
            d['why'] = r.msg
            ctx.failed.append((d['rel'], r.msg))
        else:
            ok.append(d)
    if len(ctx.keep) - len(ok) > C.max_fail_frac * len(ctx.keep):
        raise PrepError(f'{len(ctx.keep) - len(ok)} 张图写出失败，超过 max_fail_frac={C.max_fail_frac:.0%}', f'结果留在 {ctx.stage}；检查磁盘空间/权限/文件是否被占用')
    ctx.keep = ok


def rank_by_diversity(keep):
    """最远点采样：先标最有代表性（彼此差异最大）的图"""
    order = list(range(len(keep)))
    if len(keep) > 2 and C.priority_n > 0:
        X = np.hstack([np.stack([d['r']['th'] for d in keep]) / 16, np.stack([d['r']['cm'] for d in keep]) / 64])
        n_pick = min(C.priority_n, len(keep))
        pick = [int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))]
        dist = ((X - X[pick[0]]) ** 2).sum(1)
        while len(pick) < n_pick:
            j = int(np.argmax(dist))
            pick.append(j)
            dist = np.minimum(dist, ((X - X[j]) ** 2).sum(1))
        ps = set(pick)
        order = pick + [i for i in order if i not in ps]
    return order


def stage_reports(ctx, frames_dir):
    out = ctx.stage
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)
    keep = ctx.keep
    order = rank_by_diversity(keep)
    rank = {i: r + 1 for r, i in enumerate(order)}
    kidx = {id(d): i for i, d in enumerate(keep)}
    save_csv(os.path.join(out, 'quality.csv'), ['source', 'status', 'issues', 'detail', 'new_name', 'group', 'width', 'height', 'blur', 'mean_gray', 'rank', 'meta'],
             [(d['rel'], 'kept' if id(d) in kidx else 'rejected', ';'.join(d['issues']), d.get('why', ''), d.get('new', ''), d['group'],
               *(d['r'].get('eff', ('', ''))), f"{d['r'].get('blur', 0):.1f}", f"{d['r'].get('mean', 0):.0f}",
               rank[kidx[id(d)]] if id(d) in kidx else '', d['meta']) for d in ctx.items])
    save_csv(os.path.join(out, 'group_manifest.csv'), ['image', 'group'], [(os.path.splitext(d['new'])[0], d['group']) for d in keep if d['group']])
    save_csv(os.path.join(out, 'annotation_order.csv'), ['rank', 'image', 'group', 'flags'], [(rank[i], keep[i]['new'], keep[i]['group'], ';'.join(keep[i]['issues'])) for i in order])
    if ctx.failed:
        save_csv(os.path.join(out, 'failed.csv'), ['source', 'reason'], ctx.failed)
    if C.batch_size > 0:
        for b in range(0, len(order), C.batch_size):
            names = [keep[i]['new'] for i in order[b:b + C.batch_size]]
            tag = f'batch_{b // C.batch_size + 1:03d}'
            if C.batch_mode == 'copy':
                for nme in names:
                    dst = os.path.join(out, 'batches', tag, nme.replace('/', '__'))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(os.path.join(out, 'images', nme), dst)
            else:
                write_text(os.path.join(out, 'batches', tag + '.txt'), '\n'.join(names) + '\n')
    write_text(os.path.join(out, 'run_info.json'), json.dumps(dict(version=__version__, config=vars(C), kept=len(keep), total=len(ctx.items), failed=len(ctx.failed), issues=dict(ctx.cnt),
                                                                   opencv=cv.__version__, seconds=round(time.time() - ctx.t0, 1)), ensure_ascii=False, indent=1, default=str))


def stage_finalize(ctx):
    flagged = sum(1 for d in ctx.keep if d['issues'])
    log(f'[5/5] 完成：{len(ctx.keep)} 张（其中 {flagged} 张被标记，见 quality.csv）；用时 {time.time() - ctx.t0:.1f}s')
    close_log()
    try:
        if ctx.replace_existing:
            shutil.rmtree(ctx.out)
        os.replace(ctx.stage, ctx.out)
    except OSError as e:
        raise OutputError(f'把 {ctx.stage} 改名为 {ctx.out} 失败：{e}', (os_error_hint(e) or '关闭占用该文件夹的程序后，手动把 .partial 改名即可') + f'；数据已完整生成在 {ctx.stage}')
    print(f'图片在 {os.path.join(ctx.out, "images")}')
    print(f'下一步：导入标注工具标注；标完后运行"标注后"脚本，加 --manifest {os.path.join(ctx.out, "group_manifest.csv")}')


def write_crash_log(path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'标注前-素材整理 {__version__} 崩溃日志\n时间 {time.strftime("%Y-%m-%d %H:%M:%S")}\nPython {sys.version}\nOpenCV {cv.__version__} NumPy {np.__version__}\n平台 {sys.platform}\n命令行 {sys.argv}\n\n'
                    f'配置:\n{json.dumps(vars(C), ensure_ascii=False, indent=1, default=str)}\n\n最近日志:\n' + '\n'.join(LOGBUF) + '\n\n' + traceback.format_exc())
        return path
    except Exception:                                               # noqa
        return None


def run():
    ctx = new_context()
    stage_prepare(ctx)
    vids = stage_collect(ctx)
    frames_dir = stage_extract(ctx, vids)
    stage_scan_filter(ctx)
    stage_write_images(ctx)
    stage_reports(ctx, frames_dir)
    stage_finalize(ctx)


def main():
    try:
        run()
    except KeyboardInterrupt:
        log(f'\n已被用户中断。未完成的结果留在 {C.out_dir}.partial（已有的 {C.out_dir} 没被改动）。')
        close_log()
        sys.exit(130)
    except PrepError as e:
        log(f'\n✖ {e}' + (f'\n  建议：{e.hint}' if e.hint else ''))
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

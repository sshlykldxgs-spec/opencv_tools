import os
import json
import collections
import xml.etree.ElementTree as ET

# 标注工具导出的格式各不相同，YOLO 要的是：每张图一个 txt，每行 "类别编号 xc yc w h"（0~1 归一化）。
# CVAT 可以直接导出 YOLO；这里处理另外两种常见来源：Pascal VOC(xml) 和 COCO(json)。

VOC_DIR = 'annotations_xml'     # VOC：一堆 xml
COCO_JSON = None                # COCO：一个 json 文件路径（用 COCO 时填上，VOC_DIR 设 None）
OUT_DIR = 'labels'
CLASSES = None                  # 类别名列表，决定类别编号顺序，如 ['cat', 'dog']；VOC 时 None = 按名字排序自动生成


def box_to_yolo(x1, y1, x2, y2, w, h):
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def voc_to_yolo(xml_dir, out_dir, classes=None):
    """VOC 的框是像素坐标 (xmin,ymin,xmax,ymax)，且是 1 起始；转成 0~1 归一化的中心点+宽高。
    classes 为 None 时先扫一遍所有 xml 收集类别（按名字排序）。返回类别列表。"""
    files = sorted(f for f in os.listdir(xml_dir) if f.endswith('.xml'))
    parsed = []
    for f in files:
        root = ET.parse(os.path.join(xml_dir, f)).getroot()
        size = root.find('size')
        w, h = int(float(size.find('width').text)), int(float(size.find('height').text))
        objs = []
        for o in root.findall('object'):
            bb = o.find('bndbox')
            objs.append((o.find('name').text.strip(), *[float(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]))
        parsed.append((os.path.splitext(f)[0], w, h, objs))
    if classes is None:
        classes = sorted({o[0] for p in parsed for o in p[3]})
    cid = {c: i for i, c in enumerate(classes)}
    os.makedirs(out_dir, exist_ok=True)
    for stem, w, h, objs in parsed:
        with open(os.path.join(out_dir, stem + '.txt'), 'w') as fo:
            for name, x1, y1, x2, y2 in objs:
                if name not in cid:
                    continue
                xc, yc, bw, bh = box_to_yolo(x1 - 1, y1 - 1, x2, y2, w, h)
                if bw > 0 and bh > 0:
                    fo.write(f'{cid[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')
    return classes


def coco_to_yolo(json_path, out_dir):
    """COCO 的框是像素 (x, y, 宽, 高)，左上角起点；类别 id 不连续（常见 1 起或有空缺），这里重新映射成 0..N-1。
    iscrowd=1 的框跳过。返回类别名列表（下标即 YOLO 类别编号）。"""
    d = json.load(open(json_path, encoding='utf-8'))
    cats = sorted(d['categories'], key=lambda c: c['id'])
    cid = {c['id']: i for i, c in enumerate(cats)}
    imgs = {im['id']: im for im in d['images']}
    per_img = collections.defaultdict(list)
    for a in d['annotations']:
        if a.get('iscrowd', 0):
            continue
        per_img[a['image_id']].append(a)
    os.makedirs(out_dir, exist_ok=True)
    for iid, im in imgs.items():  # 没有标注的图也生成空 txt（= 纯背景图）
        stem = os.path.splitext(os.path.basename(im['file_name']))[0]
        with open(os.path.join(out_dir, stem + '.txt'), 'w') as fo:
            for a in per_img.get(iid, []):
                x, y, bw, bh = a['bbox']
                xc, yc, nw, nh = box_to_yolo(x, y, x + bw, y + bh, im['width'], im['height'])
                if nw > 0 and nh > 0:
                    fo.write(f"{cid[a['category_id']]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
    return [c['name'] for c in cats]


if __name__ == '__main__':
    if COCO_JSON:
        names = coco_to_yolo(COCO_JSON, OUT_DIR)
    elif VOC_DIR:
        names = voc_to_yolo(VOC_DIR, OUT_DIR, CLASSES)
    else:
        raise SystemExit('请在上面配置 VOC_DIR 或 COCO_JSON')
    print('转换完成 →', OUT_DIR)
    print('类别（下标就是 YOLO 的类别编号，写进 data.yaml 时要一致）：', names)

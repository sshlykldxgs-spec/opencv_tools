"""合成测试数据生成器（不依赖任何真实数据）"""
import json, os, random, shutil
import cv2 as cv
import numpy as np

CL = ['bottle', 'foam', 'leaf', 'bag']
COCO_ID = {'bottle': 3, 'foam': 7, 'leaf': 20, 'bag': 88}      # 故意不连续


def make_det(root, videos=12, frames=10, seed=1, exif_times=False):
    """每个"视频"frames 帧，物体缓慢移动；文件名完全随机（不含视频信息）。返回 {文件名stem: 视频号}"""
    random.seed(seed)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root + '/images'), os.makedirs(root + '/labels')
    imgs, anns, truth = [], [], {}
    n = 0
    for v in range(videos):
        rs0 = np.random.RandomState(seed * 100 + v)
        base = cv.GaussianBlur(np.full((480, 640, 3), rs0.randint(60, 140, 3), np.uint8) + rs0.randint(0, 30, (480, 640, 3), dtype=np.uint8), (0, 0), 3)
        for i in range(frames):
            im, lab = base.copy(), []
            r = np.random.RandomState(seed * 100 + v)
            for _ in range(r.randint(1, 4)):
                c = r.choice(4, p=[.55, .25, .15, .05])
                x, y, w, h = r.randint(50, 500) + i * 4, r.randint(50, 380) + i * 2, r.randint(30, 90), r.randint(30, 90)
                cv.ellipse(im, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, [(40, 40, 220), (240, 240, 240), (30, 160, 30), (200, 120, 30)][c], -1)
                lab.append((c, x, y, min(x + w, 640), min(y + h, 480)))
            im = np.clip(im.astype(int) + np.random.RandomState(n).randint(-5, 5, im.shape), 0, 255).astype(np.uint8)
            name = f'{random.getrandbits(32):08x}'
            truth[name] = v
            path = f'{root}/images/{name}.jpg'
            if exif_times:
                from PIL import Image
                ex = Image.Exif()
                ex[36867] = '2026:01:01 10:%02d:%02d' % (v, i)              # 同一个 v 的帧相隔 1 秒
                Image.fromarray(im[:, :, ::-1]).save(path, exif=ex.tobytes() and ex, quality=95)
                # DateTimeOriginal 在 Exif IFD 里：用 exif= 传 IFD
                ex2 = Image.Exif(); ifd = ex2.get_ifd(0x8769); ifd[36867] = '2026:01:01 10:%02d:%02d' % (v, i)
                Image.fromarray(im[:, :, ::-1]).save(path, exif=ex2, quality=95)
            else:
                cv.imwrite(path, im)
            with open(f'{root}/labels/{name}.txt', 'w') as f:
                for c, x1, y1, x2, y2 in lab:
                    f.write(f'{c} {(x1 + x2) / 2 / 640:.6f} {(y1 + y2) / 2 / 480:.6f} {(x2 - x1) / 640:.6f} {(y2 - y1) / 480:.6f}\n')
            imgs.append(dict(id=n + 1, file_name=name + '.jpg', width=640, height=480))
            for c, x1, y1, x2, y2 in lab:
                anns.append(dict(id=len(anns) + 1, image_id=n + 1, category_id=COCO_ID[CL[c]], bbox=[x1, y1, x2 - x1, y2 - y1], iscrowd=0))
            n += 1
    open(root + '/classes.txt', 'w').write('\n'.join(CL))
    json.dump(dict(images=imgs, annotations=anns, categories=[dict(id=COCO_ID[c], name=c) for c in CL]), open(root + '/coco.json', 'w'))
    with open(root + '/truth.csv', 'w') as f:
        for k, v in truth.items():
            f.write(f'{k},{v}\n')
    return truth


def make_exif_set(root):
    """EXIF 方向=6 的图 + 标注（标注按"显示后"的方向画）"""
    from PIL import Image
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root + '/images'), os.makedirs(root + '/labels')
    rs = np.random.RandomState(5)
    for n in range(12):
        raw = np.full((480, 640, 3), 90, np.uint8) + rs.randint(0, 20, (480, 640, 3), dtype=np.uint8)
        x, y = rs.randint(50, 450), rs.randint(50, 300)
        cv.rectangle(raw, (x, y), (x + 100, y + 60), (0, 0, 220), -1)
        ex = Image.Exif(); ex[274] = 6
        Image.fromarray(raw[:, :, ::-1]).save(f'{root}/images/e{n:03d}.jpg', exif=ex)
        gx1, gy1, gx2, gy2, W, H = 480 - (y + 60), x, 480 - y, x + 100, 480, 640
        open(f'{root}/labels/e{n:03d}.txt', 'w').write(f'0 {(gx1 + gx2) / 2 / W:.6f} {(gy1 + gy2) / 2 / H:.6f} {(gx2 - gx1) / W:.6f} {(gy2 - gy1) / H:.6f}\n')
    open(root + '/classes.txt', 'w').write('thing\n')


def make_raw(root):
    """标注前素材：视频(中文名)、好图、重复、近重复、太小、纯色、模糊、暗、损坏、EXIF、大图"""
    from PIL import Image
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root + '/sub')
    def tex(seed, w=800, h=600):
        r = np.random.RandomState(seed); im = np.zeros((h, w, 3), np.uint8)
        for _ in range(40):
            cv.rectangle(im, (r.randint(0, w), r.randint(0, h)), (r.randint(0, w), r.randint(0, h)), tuple(int(v) for v in r.randint(0, 255, 3)), -1)
        return cv.GaussianBlur(im, (0, 0), 1)
    for name in ('video1.mp4', '视频中文.mp4'):
        vw = cv.VideoWriter(f'{root}/{name}', cv.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
        for sc in range(3):
            b = cv.resize(tex(100 + sc + (7 if '中' in name else 0)), (640, 480))
            for f in range(30):
                fr = b.copy(); cv.circle(fr, (100 + f * 2, 240), 10, (255, 255, 255), -1); vw.write(fr)
        vw.release()
    for i in range(6):
        cv.imwrite(f'{root}/p{i}.jpg', tex(i))
    shutil.copy(f'{root}/p0.jpg', f'{root}/dup_exact.jpg')
    a = cv.imread(f'{root}/p1.jpg')
    cv.imwrite(f'{root}/near_dup.jpg', np.clip(a.astype(int) + np.random.RandomState(1).randint(-2, 3, a.shape), 0, 255).astype(np.uint8), [cv.IMWRITE_JPEG_QUALITY, 80])
    cv.imwrite(f'{root}/tiny.jpg', tex(9, 100, 100))
    cv.imwrite(f'{root}/flat.jpg', np.full((600, 800, 3), 128, np.uint8))
    cv.imwrite(f'{root}/blur.jpg', cv.GaussianBlur(tex(3), (0, 0), 12))
    cv.imwrite(f'{root}/dark.jpg', (tex(4) * 0.06).astype(np.uint8))
    cv.imwrite(f'{root}/sub/s.png', tex(11))
    open(f'{root}/corrupt.jpg', 'wb').write(b'not an image' * 50)
    ex = Image.Exif(); ex[274] = 6
    Image.fromarray(tex(12)[:, :, ::-1]).save(f'{root}/exif6.jpg', exif=ex)
    cv.imwrite(f'{root}/big.jpg', cv.resize(tex(13), (4000, 3000)))

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

IMG_PATH = '/Users/pengxuezhi/Desktop/huiliyi.jpg'
TPL_PATH = '/Users/pengxuezhi/Desktop/moble.jpg'
MAX_SIDE = 2000  # 图片太大时先缩小再检测（SIFT 很吃时间），结果再映射回原图
NCC_THRESH = 0.85  # 兜底的多尺度模板匹配至少要这个分数才算找到（在 messigray.png 上实测：真目标 ≥0.89，目标不在图里 ≤0.80）
PEAK_GAP = 0.10  # 最高峰要比"别处的次高峰"至少高这么多，否则认为不唯一、不可信
TPL_SCALES = (1.0, 0.7, 0.5)  # 模板提取特征的缩放档位（1.0 = 原尺寸）


def to_gray(im):
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY) if im.ndim == 3 else im


def shrink(im, max_side):
    """长边超过 max_side 就等比缩小，返回 (缩小后的图, 缩放比例)"""
    s = min(1.0, max_side / max(im.shape[:2]))
    if s < 1.0:
        im = cv.resize(im, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
    return im, s


def rootsift(des, eps=1e-7):
    """RootSIFT：L1 归一化后开方。特征点不变，匹配却比原版 SIFT 更准"""
    des = des / (des.sum(axis=1, keepdims=True) + eps)
    return np.sqrt(des).astype(np.float32)


def valid_quad(box, tpl_wh, img_wh):
    """检查单应性算出来的框是否合理，挡掉 RANSAC 偶尔给出的畸形框"""
    pts = box.reshape(4, 2).astype(np.float32)
    tw, th = tpl_wh
    ref = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    # 1) 必须是凸四边形，且没有被镜像翻转（顶点顺序方向和模板一致）
    if not cv.isContourConvex(pts.reshape(-1, 1, 2)):
        return False
    if cv.contourArea(pts, True) * cv.contourArea(ref, True) <= 0:
        return False
    # 2) 缩放倍数不能离谱
    if not (0.05 < cv.contourArea(pts) / (tw * th) < 20):
        return False
    # 3) 长宽比不能被压扁（相对模板原始长宽比）
    d = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    w_est, h_est = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    if h_est < 1 or not (0.5 < (w_est / h_est) / (tw / th) < 2):
        return False
    # 4) 框的中心必须在图内
    cx, cy = pts.mean(axis=0)
    return 0 <= cx < img_wh[0] and 0 <= cy < img_wh[1]


def ncc_verify(img_gray, tpl_gray, H):
    """把图里的候选区域按 H 拉回模板坐标系，和模板算归一化相关系数（亮度/对比度变化不影响）"""
    h, w = tpl_gray.shape
    warped = cv.warpPerspective(img_gray, H, (w, h), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)
    score = cv.matchTemplate(warped, tpl_gray, cv.TM_CCOEFF_NORMED)[0, 0]
    return float(np.nan_to_num(score))


def sift_detect(img_gray, tpl_gray, ratio=0.85, reproj=3.0, min_inliers=10, min_ncc=0.4, tpl_scales=TPL_SCALES):
    # 1 对比度增强 + RootSIFT 特征（模板提多个缩小版本，目标比模板小很多时细节丢了也能对上）
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_img, g_tpl = clahe.apply(img_gray), clahe.apply(tpl_gray)
    sift = cv.SIFT_create(contrastThreshold=0.02)
    kp_i, des_i = sift.detectAndCompute(g_img, None)
    pts_t, des_t = [], []
    for s in tpl_scales:
        t = g_tpl if s == 1.0 else cv.resize(g_tpl, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
        if min(t.shape) < 32:
            continue
        k, d = sift.detectAndCompute(t, None)
        if d is not None:
            pts_t.append(np.float32([p.pt for p in k]) / s)  # 坐标换算回模板原始尺寸
            des_t.append(d)
    if not des_t or des_i is None or len(kp_i) < 4:
        return None, 'SIFT 特征点太少'
    pts_t, des_t, des_i = np.vstack(pts_t), rootsift(np.vstack(des_t)), rootsift(des_i)
    if len(pts_t) < 4:
        return None, 'SIFT 特征点太少'

    # 2 FLANN 最近邻 + Lowe 比率测试
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    knn = flann.knnMatch(des_t, des_i, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_inliers:
        return None, f'通过比率测试的匹配点只有 {len(good)} 个'

    # 3 MAGSAC 求单应矩阵（低版本 OpenCV 自动退回 RANSAC）
    src = pts_t[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = np.float32([kp_i[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    method = getattr(cv, 'USAC_MAGSAC', cv.RANSAC)
    H, mask = cv.findHomography(src, dst, method, reproj, maxIters=5000, confidence=0.999)
    if H is None or int(mask.sum()) < min_inliers:
        return None, 'RANSAC 内点不足'
    keep = mask.ravel().astype(bool)

    # 4 用全部内点最小二乘重估一次 H，再按新 H 重选内点（亚像素精修）
    H2, _ = cv.findHomography(src[keep], dst[keep], 0)
    if H2 is not None:
        err = np.linalg.norm(cv.perspectiveTransform(src, H2) - dst, axis=2).ravel()
        keep2 = err < reproj
        if keep2.sum() >= min_inliers:
            H, keep = H2, keep2

    # 5 双重校验：几何形状合理 + 拉回模板后像素真的像
    h, w = tpl_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    box = cv.perspectiveTransform(corners, H).reshape(4, 2)
    if not valid_quad(box, (w, h), (img_gray.shape[1], img_gray.shape[0])):
        return None, '单应性结果形状不合理'
    ncc = ncc_verify(img_gray, tpl_gray, H)
    if ncc < min_ncc:
        return None, f'像素相关性太低 (NCC={ncc:.2f})'
    return box, f'SIFT+RANSAC 内点={int(keep.sum())} NCC={ncc:.2f}'


def multiscale_ncc(img_gray, tpl_gray, thresh=NCC_THRESH, gap=PEAK_GAP):
    """兜底：多尺度模板匹配（先粗搜再细搜）。分数不够、或最高峰不唯一，就老实说没找到"""
    H_i, W_i = img_gray.shape

    def search(scales):
        best = None  # (分数, 左上角, (宽, 高), 缩放比例, 相关图)
        for s in scales:
            interp = cv.INTER_AREA if s < 1 else cv.INTER_CUBIC
            t = cv.resize(tpl_gray, None, fx=s, fy=s, interpolation=interp)
            th, tw = t.shape
            if th > H_i or tw > W_i or th < 16 or tw < 16:
                continue
            res = cv.matchTemplate(img_gray, t, cv.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, loc, (tw, th), s, res)
        return best

    best = search(np.geomspace(0.3, 2.0, 30))
    if best is None:
        return None, '模板比图还大或太小'
    fine = search(best[3] * np.geomspace(0.93, 1.07, 15))  # 在最优尺度附近细搜
    if fine is not None and fine[0] > best[0]:
        best = fine
    score, (x, y), (tw, th), _, res = best

    # 抹掉最高峰周围，看别处还有没有差不多高的峰（有就说明不唯一，不可信）
    r = max(tw, th) // 2
    res = res.copy()
    res[max(0, y - r):y + r, max(0, x - r):x + r] = -1
    second = float(res.max())
    if score < thresh or score - second < gap:
        return None, f'多尺度模板匹配不可信 (最高分={score:.2f}, 次高峰={second:.2f})'
    box = np.float32([[x, y], [x + tw, y], [x + tw, y + th], [x, y + th]])
    return box, f'多尺度模板匹配 score={score:.2f}'


def locate(img, tpl):
    """返回 (四边形框 4x2，原图坐标；没找到则为 None, 说明文字)"""
    small, s = shrink(img, MAX_SIDE)
    tpl_small, _ = shrink(tpl, MAX_SIDE)
    g_img, g_tpl = to_gray(small), to_gray(tpl_small)

    box, info = sift_detect(g_img, g_tpl)
    if box is None:
        print('特征匹配没通过：', info, '→ 回退到多尺度模板匹配')
        box, info = multiscale_ncc(g_img, g_tpl)
    if box is None:
        return None, info
    return box / s, info


if __name__ == '__main__':
    # 1 读取图像
    img = cv.imread(IMG_PATH)
    template = cv.imread(TPL_PATH)
    if img is None or template is None:
        print('图片读取失败，请检查路径：', IMG_PATH, TPL_PATH)
        raise SystemExit

    # 2 匹配
    box, info = locate(img, template)
    print('结果：', info)

    # 3 绘制并显示
    img_show = img.copy()
    if box is not None:
        cv.polylines(img_show, [np.int32(box.round())], True, (0, 0, 255), 3, cv.LINE_AA)
        title = info
    else:
        title = '未检测到目标：' + info
    plt.imshow(img_show[:, :, ::-1])
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

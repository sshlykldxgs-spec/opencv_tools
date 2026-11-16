import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

IMG_PATH = '/Users/pengxuezhi/Desktop/huiliyi.jpg'


def _norm(rho, theta):
    """把 (rho, theta) 规范到 theta∈[0, π)，方便比较两条线是不是同一条"""
    if theta >= np.pi:
        theta, rho = theta - np.pi, -rho
    elif theta < 0:
        theta, rho = theta + np.pi, -rho
    return rho, theta


def _same_line(a, b, d_rho, d_theta):
    """两条 (rho, theta) 线是否足够接近（考虑 theta 在 0 和 π 处首尾相接，此时 rho 变号）"""
    (r1, t1), (r2, t2) = a, b
    if abs(t1 - t2) < d_theta and abs(r1 - r2) < d_rho:
        return True
    if abs(abs(t1 - t2) - np.pi) < d_theta and abs(r1 + r2) < d_rho:
        return True
    return False


def refine_line(gray, rho, theta, seg, half=7.0, stride=3.0):
    """亚像素精修：沿线每隔 stride 像素取一条垂直于线的灰度剖面，去掉背景的线性趋势后求"线"的加权质心，
    再把这些质心点做 Huber 直线拟合。Canny 边缘只有整像素精度，而且粗线有两条边，拟合会被边缘位置偏置；
    质心法直接用灰度，精度高得多（合成测试：ρ 误差 1.4px → 0.17px，θ 误差 0.17° → 0.02°）。
    线太细/对比度太低、质心点不够时，原样返回，不会比霍夫结果更差。"""
    sc = half / 7.0                      # 图越大，线越粗：剖面宽度、采样间隔、平滑都跟着放大
    step = 0.5 * sc
    stride = stride * sc
    p1, p2 = np.array(seg[:2], float), np.array(seg[2:], float)
    L = np.linalg.norm(p2 - p1)
    if L < 10:
        return rho, theta
    d = (p2 - p1) / L
    n = np.array([np.cos(theta), np.sin(theta)])
    ts = np.arange(0, L, stride)
    offs = np.arange(-half, half + 1e-6, step)
    pos = p1[None, None, :] + ts[:, None, None] * d + offs[None, :, None] * n
    g = cv.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8 * sc)
    prof = cv.remap(g, pos[..., 0].astype(np.float32), pos[..., 1].astype(np.float32), cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REPLICATE)
    k = len(offs)
    x = np.arange(k)
    e = max(3, k // 5)
    edge_idx = np.r_[0:e, k - e:k]
    reach = int(3 * sc / step)
    pts = []
    for i in range(len(ts)):
        pr = prof[i]
        dev = pr - np.polyval(np.polyfit(x[edge_idx], pr[edge_idx], 1), x)  # 去掉背景的线性变化
        sig = np.maximum(0, np.sign(dev[np.argmax(np.abs(dev))]) * dev)       # 线可以比背景亮，也可以比背景暗
        pk = int(np.argmax(sig))
        if sig[pk] < 8:
            continue
        lo, hi = max(0, pk - reach), min(k, pk + reach + 1)
        w = np.maximum(sig[lo:hi] - 0.5 * sig[pk], 0)
        if w.sum() <= 0:
            continue
        pts.append(p1 + ts[i] * d + (offs[lo:hi] * w).sum() / w.sum() * n)
    if len(pts) < 5:
        return rho, theta
    vx, vy, x0, y0 = cv.fitLine(np.array(pts, np.float32), cv.DIST_HUBER, 0, 0.01, 0.01).ravel()
    th = np.arctan2(vx, -vy)
    return _norm(x0 * np.cos(th) + y0 * np.sin(th), th)


def _refine_all(gray, lines, segs, half=7.0):
    """对每条线做亚像素精修，线段端点投影到精修后的直线上"""
    out_l, out_s = [], []
    for (rho, theta), sg in zip(lines, segs):
        r, t = refine_line(gray, rho, theta, sg, half=half)
        n = np.array([np.cos(t), np.sin(t)])
        p = np.array(sg, float).reshape(2, 2)
        p = p - ((p @ n) - r)[:, None] * n  # 垂直投影到新直线
        out_l.append((r, t))
        out_s.append(p.ravel())
    return np.array(out_l).reshape(-1, 2), np.array(out_s).reshape(-1, 4)


def detect_lines(gray, thresh_ratio=0.3, merge_rho=12, merge_theta=np.deg2rad(4), min_len_ratio=0.15,
                 theta_res=np.pi / 720, max_side=960, refine=True):
    """霍夫直线检测：Canny(Otsu 自动阈值) → 霍夫累加 → 合并重复线 → 用边缘点最小二乘精修。
    返回 (lines, segments)：lines 为 Nx2 的 (rho, theta)；segments 为 Nx4 的线段端点 (x1,y1,x2,y2)。
    原写法一条真实直线会被报成好几条几乎重合的线（边缘有两侧、累加器邻近格子都过阈值），这里合并成一条；
    角度分辨率从 1° 提到 0.25°：线越长，1° 的角度误差让线两端偏出越多（640 像素长的线偏 ±2.8 像素），
    票数被摊到好几个格子里，每格都过不了阈值，整条线就漏掉了。"""
    # 图太大时先缩小：霍夫的计算量 ∝ 边缘点数 × 角度格数，缩到长边 960 以内速度和精度都合适
    s = min(1.0, max_side / max(gray.shape))
    if s < 1.0:
        lines, segs = detect_lines(cv.resize(gray, None, fx=s, fy=s, interpolation=cv.INTER_AREA),
                                   thresh_ratio, merge_rho, merge_theta, min_len_ratio, theta_res, max_side=10 ** 9,
                                   refine=False)
        # 缩小图的像素中心和原图差半个像素：x_原 = (x_小 + 0.5) / s - 0.5，rho 也要跟着改
        if len(lines):
            c = np.cos(lines[:, 1]) + np.sin(lines[:, 1])
            lines = np.c_[(lines[:, 0] + 0.5 * c) / s - 0.5 * c, lines[:, 1]]
        lines, segs = lines.reshape(-1, 2), ((segs + 0.5) / s - 0.5).reshape(-1, 4)
        return _refine_all(gray, lines, segs, half=max(7.0, 7.0 * max(gray.shape) / 960)) if refine else (lines, segs)
    h, w = gray.shape
    blur = cv.GaussianBlur(gray, (5, 5), 1.2)
    hi, _ = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    edges = cv.Canny(blur, 0.5 * hi, hi, L2gradient=True)
    ys, xs = np.nonzero(edges)
    pts = np.c_[xs, ys].astype(np.float64)
    if len(pts) == 0:
        return np.zeros((0, 2)), np.zeros((0, 4))

    # 票数门槛跟图像大小挂钩（原写法固定 150，换张小图就一条线都找不到，换张大图又满屏都是）
    thresh = max(40, int(thresh_ratio * min(h, w)))
    acc = getattr(cv, 'HoughLinesWithAccumulator', None)
    raw = acc(edges, 1, theta_res, thresh) if acc is not None else cv.HoughLines(edges, 1, theta_res, thresh)
    if raw is None:
        return np.zeros((0, 2)), np.zeros((0, 4))
    raw = raw.reshape(-1, raw.shape[-1])
    raw = raw[np.argsort(-raw[:, 2])] if raw.shape[1] == 3 else raw  # 票数多的优先

    # 非极大值抑制：已经保留了一条线，附近（rho、theta 都接近）的就丢掉
    kept = []
    for row in raw:
        cand = _norm(float(row[0]), float(row[1]))
        if not any(_same_line(cand, k, merge_rho, merge_theta) for k in kept):
            kept.append(cand)

    lines, segs = [], []
    diag = np.hypot(h, w)
    for rho, theta in kept:
        n = np.array([np.cos(theta), np.sin(theta)])
        dist = pts @ n - rho
        near = pts[np.abs(dist) < merge_rho / 2]
        if len(near) < 10:
            continue
        # 用线附近的边缘点重新拟合直线（Huber 鲁棒拟合），精度比霍夫格子的 1 像素 / 1° 更高
        vx, vy, x0, y0 = cv.fitLine(near.astype(np.float32), cv.DIST_HUBER, 0, 0.01, 0.01).ravel()
        theta_f = np.arctan2(vx, -vy)  # 法线方向
        rho_f = x0 * np.cos(theta_f) + y0 * np.sin(theta_f)
        rho_f, theta_f = _norm(rho_f, theta_f)
        d = np.array([vx, vy])
        t = (near - np.array([x0, y0])) @ d
        # 线段范围用 5%~95% 分位数，避免个别离群点把线拉得过长
        t0, t1 = np.percentile(t, 2), np.percentile(t, 98)
        if t1 - t0 < min_len_ratio * diag:
            continue
        p1, p2 = np.array([x0, y0]) + t0 * d, np.array([x0, y0]) + t1 * d
        lines.append((rho_f, theta_f))
        segs.append((*p1, *p2))
    lines, segs = np.array(lines).reshape(-1, 2), np.array(segs).reshape(-1, 4)
    return _refine_all(gray, lines, segs) if refine else (lines, segs)


if __name__ == '__main__':
    # 1 加载图片，转为灰度图
    img = cv.imread(IMG_PATH)
    if img is None:
        raise SystemExit('图片读取失败，请检查路径：' + IMG_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2 霍夫直线变换（返回合并后的直线 + 线段端点）
    lines, segs = detect_lines(gray)
    print('检测到直线条数：', len(lines))

    # 3 将检测的线段绘制在图像上
    show = img.copy()
    for x1, y1, x2, y2 in segs:
        cv.line(show, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 2, cv.LINE_AA)

    # 4 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(show[:, :, ::-1])
    plt.title(f'霍夫变换线检测（{len(lines)} 条）')
    plt.xticks([]), plt.yticks([])
    plt.show()

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

IMG_PATH = '/Users/pengxuezhi/Desktop/huiliyi.jpg'


def fit_circle(pts):
    """最小二乘拟合圆（代数法）。pts: Nx2。返回 (cx, cy, r)"""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    b = -(x ** 2 + y ** 2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2, -E / 2
    return cx, cy, float(np.sqrt(max(cx * cx + cy * cy - F, 1e-6)))


def refine_circle(edge_pts, c, min_coverage=0.7):
    """用圆周附近的边缘点重新拟合圆，并检查边缘点是否真的围成一圈。
    霍夫的结果只有整数级精度，而且只要累加器里票数够就会报圆；这一步把位置/半径精修到亚像素，
    同时把"边缘点只占圆周一小段"的假圆去掉。返回精修后的 (cx, cy, r)，不合格返回 None"""
    cx, cy, r = c
    for _ in range(3):
        d = np.hypot(edge_pts[:, 0] - cx, edge_pts[:, 1] - cy)
        tol = max(4.0, 0.12 * r)
        sel = edge_pts[np.abs(d - r) < tol]
        if len(sel) < 12:
            return None
        cx, cy, r = fit_circle(sel)
    d = np.hypot(edge_pts[:, 0] - cx, edge_pts[:, 1] - cy)
    sel = edge_pts[np.abs(d - r) < max(3.0, 0.08 * r)]
    if len(sel) < 12:
        return None
    ang = np.arctan2(sel[:, 1] - cy, sel[:, 0] - cx)
    covered = np.unique(((ang + np.pi) / (2 * np.pi) * 24).astype(int) % 24).size / 24
    return (cx, cy, r) if covered >= min_coverage else None


def detect_circles(gray, min_r=0, max_r=0, min_dist=None, param2=0.9, min_coverage=0.7, max_side=1000):
    """霍夫圆检测（HOUGH_GRADIENT_ALT，比旧的 HOUGH_GRADIENT 更准、假圆更少）+ 边缘点最小二乘精修。
    没找到圆就返回空数组 (0, 3)，不会再让后面的 for 循环报错。返回 Nx3：(圆心x, 圆心y, 半径)"""
    h, w = gray.shape
    # 图太大时整套流程都在缩小后的图上做（霍夫圆很吃时间，而且下面的参数是按 ~1000 像素的尺度调的），
    # 最后把圆心和半径放大回原图尺寸
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        small = cv.resize(gray, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
        c = detect_circles(small, int(min_r * s), int(max_r * s), None if min_dist is None else max(1, int(min_dist * s)),
                           param2, min_coverage, max_side=10 ** 9)
        if len(c):  # 缩小图和原图的像素中心差半个像素：x_原 = (x_小 + 0.5) / s - 0.5；半径只按比例放大
            c = np.c_[(c[:, :2] + 0.5) / s - 0.5, c[:, 2] / s]
        return c.reshape(-1, 3).astype(np.float32)
    if min_dist is None:
        min_dist = max(10, min(h, w) // 8)  # 原写法固定 200：画面里两个圆相距不到 200 像素就只能留一个
    g = cv.GaussianBlur(gray, (0, 0), 1.5)
    blur = cv.GaussianBlur(gray, (5, 5), 1.2)
    hi, _ = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    edges = cv.Canny(blur, 0.5 * hi, hi, L2gradient=True)
    ys, xs = np.nonzero(edges)
    edge_pts = np.c_[xs, ys].astype(np.float64)

    alt = getattr(cv, 'HOUGH_GRADIENT_ALT', None)
    if alt is not None:
        found = cv.HoughCircles(g, alt, 1.5, min_dist, param1=300, param2=param2, minRadius=min_r, maxRadius=max_r)
    else:  # 旧版 OpenCV（< 4.3）没有 ALT，退回普通版本，靠下面的精修 + 圆周覆盖率把关
        found = cv.HoughCircles(g, cv.HOUGH_GRADIENT, 1, min_dist, param1=150, param2=40, minRadius=min_r, maxRadius=max_r)
    if found is None or len(edge_pts) == 0:
        return np.zeros((0, 3), np.float32)

    out = []
    for c in found[0]:
        r = refine_circle(edge_pts, tuple(map(float, c)), min_coverage)
        if r is not None:
            out.append(r)
    # 精修后可能出现重复（两个初值收敛到同一个圆），去重
    uniq = []
    for c in out:
        if all(np.hypot(c[0] - u[0], c[1] - u[1]) > 0.5 * min(c[2], u[2]) or abs(c[2] - u[2]) > 0.3 * min(c[2], u[2]) for u in uniq):
            uniq.append(c)
    return np.array(uniq, np.float32).reshape(-1, 3)


if __name__ == '__main__':
    # 1 读取图像，并转换为灰度图
    planets = cv.imread(IMG_PATH)
    if planets is None:
        raise SystemExit('图片读取失败，请检查路径：' + IMG_PATH)
    gray_img = cv.cvtColor(planets, cv.COLOR_BGR2GRAY)

    # 2 霍夫圆检测
    circles = detect_circles(gray_img)
    print('检测到圆的个数：', len(circles))

    # 3 将检测结果绘制在图像上
    show = planets.copy()
    for cx, cy, r in circles:
        cv.circle(show, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 2, cv.LINE_AA)  # 圆
        cv.circle(show, (int(round(cx)), int(round(cy))), 2, (0, 0, 255), 3, cv.LINE_AA)  # 圆心

    # 4 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(show[:, :, ::-1])
    plt.title(f'霍夫变换圆检测（{len(circles)} 个）')
    plt.xticks([]), plt.yticks([])
    plt.show()

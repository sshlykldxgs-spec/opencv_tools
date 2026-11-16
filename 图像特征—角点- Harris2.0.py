import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

IMG_PATH = '/Users/pengxuezhi/Desktop/huiliyi.jpg'


def estimate_noise(gray):
    """快速估计图像噪声标准差（Immerkaer 方法，灰度 0~255）"""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray.shape
    s = np.abs(cv.filter2D(gray.astype(np.float32), -1, k))[1:-1, 1:-1].sum()
    return float(s * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def harris_corners(gray, block=3, ksize=3, k=0.04, rel_thresh=0.01, min_dist=8, refine=True):
    """Harris 角点：按噪声自适应去噪 → 响应图 → 阈值 + 非极大值抑制 → 亚像素精修。
    返回 Nx2 的 (x, y)，按响应从强到弱"""
    # 去噪强度跟着噪声走：干净的图只轻微平滑（模糊越狠，角点定位越不准），噪声大时才加强
    sigma = float(np.clip(0.8 + 0.06 * max(0.0, estimate_noise(gray) - 12), 0.8, 2.0))
    g = cv.GaussianBlur(gray, (0, 0), sigma)
    R = cv.cornerHarris(np.float32(g), block, ksize, k)

    # 非极大值抑制：只保留 (2*min_dist+1)² 邻域里响应最大的点
    # （原写法直接把 R>阈值 的所有像素涂红，每个角点会变成一团，而且阈值 0.001 太低，噪声也被选进来）
    local_max = cv.dilate(R, np.ones((2 * min_dist + 1, 2 * min_dist + 1), np.uint8))
    ys, xs = np.nonzero((R >= local_max) & (R > rel_thresh * R.max()))
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    order = np.argsort(-R[ys, xs])
    pts = np.float32(np.c_[xs[order], ys[order]])

    if refine:  # 亚像素精修：把整数坐标细化到 0.1 像素量级
        crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        pts = cv.cornerSubPix(g, pts.reshape(-1, 1, 2).copy(), (5, 5), (-1, -1), crit).reshape(-1, 2)
    return pts


if __name__ == '__main__':
    # 1 读取图像，并转换成灰度图像
    img = cv.imread(IMG_PATH)
    if img is None:
        raise SystemExit('图片读取失败，请检查路径：' + IMG_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2 角点检测
    pts = harris_corners(gray)
    print('检测到角点数：', len(pts))

    # 3 绘制角点（画小圆，不去改原图的像素）
    show = img.copy()
    for x, y in pts:
        cv.circle(show, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, cv.LINE_AA)

    # 4 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(show[:, :, ::-1])
    plt.title(f'Harris角点检测（{len(pts)} 个，已非极大值抑制 + 亚像素精修）')
    plt.xticks([]), plt.yticks([])
    plt.show()

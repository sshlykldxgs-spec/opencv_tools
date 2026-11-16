import numpy as np
import cv2 as cv
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


def shi_tomasi_corners(gray, max_corners=1000, quality=0.05, min_dist=10, block=5, refine=True):
    """Shi-Tomasi 角点：按噪声自适应去噪 → goodFeaturesToTrack → 亚像素精修。返回 Nx2 的 (x, y)
    quality：相对最强角点的比例，越小点越多（原写法 0.01 在有噪声时会选进大量假角点）"""
    # 去噪强度跟着噪声走：干净的图只轻微平滑（模糊越狠，角点定位越不准），噪声大时才加强
    sigma = float(np.clip(0.8 + 0.06 * max(0.0, estimate_noise(gray) - 12), 0.8, 2.0))
    g = cv.GaussianBlur(gray, (0, 0), sigma)
    corners = cv.goodFeaturesToTrack(g, max_corners, quality, min_dist, blockSize=block)
    if corners is None:  # 原写法这里会直接报 TypeError（纯色图、过暗的图都可能检测不到角点）
        return np.zeros((0, 2), np.float32)
    if refine:  # 亚像素精修：goodFeaturesToTrack 只给整数像素坐标
        crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv.cornerSubPix(g, corners, (5, 5), (-1, -1), crit)
    return corners.reshape(-1, 2)


if __name__ == '__main__':
    # 1 读取图像
    img = cv.imread(IMG_PATH)
    if img is None:
        raise SystemExit('图片读取失败，请检查路径：' + IMG_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2 Shi-Tomasi 角点检测
    pts = shi_tomasi_corners(gray)
    print('检测到角点数：', len(pts))

    # 3 绘制角点（画在副本上，不改动原图）
    show = img.copy()
    for x, y in pts:
        cv.circle(show, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, cv.LINE_AA)

    # 4 图像展示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(show[:, :, ::-1])
    plt.title(f'shi-tomasi角点检测（{len(pts)} 个，已亚像素精修）')
    plt.xticks([]), plt.yticks([])
    plt.show()

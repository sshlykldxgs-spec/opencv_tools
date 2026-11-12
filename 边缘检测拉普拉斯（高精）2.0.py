import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1 读取图像
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg', 0)
if img is None:
    raise SystemExit('图片读取失败')

# 2 原写法：直接 Laplacian + 取绝对值
result = cv.Laplacian(img, cv.CV_16S)
old = cv.convertScaleAbs(result)

# 3 优化写法：LoG（先高斯、再拉普拉斯）+ 零交叉，得到单像素宽的细边缘
# 3.1 先高斯平滑：二阶导会把噪声放大很多，必须先平滑
blur = cv.GaussianBlur(img, (0, 0), 1.5)
# 3.2 用浮点数保留正负号（convertScaleAbs 取绝对值后，正负号这个关键信息就丢了）
lap = cv.Laplacian(blur, cv.CV_32F, ksize=3)
# 3.3 零交叉：边缘就在 Laplacian 由正变负的地方
k = np.ones((3, 3), np.uint8)
lo, hi = cv.erode(lap, k), cv.dilate(lap, k)  # 3x3 邻域内的最小值 / 最大值
slope = hi - lo  # 邻域内正负落差
thr = 0.25 * np.percentile(slope, 99)  # 落差太小的是平坦区的抖动，去掉
edge = (((lap >= 0) & (lo < 0) & (slope > thr)) * 255).astype(np.uint8)

# 4 图像展示
plt.figure(figsize=(15, 5), dpi=100)
plt.subplot(131), plt.imshow(img, cmap=plt.cm.gray), plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(old, cmap=plt.cm.gray), plt.title('原写法（直接Laplacian）')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(edge, cmap=plt.cm.gray), plt.title('优化后（LoG + 零交叉）')
plt.xticks([]), plt.yticks([])
plt.show()

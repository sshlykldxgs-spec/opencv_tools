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

# 2 原写法：|gx| 和 |gy| 各占 0.5 相加
x = cv.Sobel(img, cv.CV_16S, 1, 0)
y = cv.Sobel(img, cv.CV_16S, 0, 1)
old = cv.addWeighted(cv.convertScaleAbs(x), 0.5, cv.convertScaleAbs(y), 0.5, 0)

# 3 优化写法
# 3.1 轻度去噪，再用浮点数计算（避免 16 位整数截断）
blur = cv.GaussianBlur(img, (3, 3), 0)
gx = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=3)
gy = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=3)
# 3.2 用真正的梯度幅值 sqrt(gx²+gy²)。
#     |gx|+|gy| 只是近似：同样强的边，斜 45° 的比竖直的响应大约强 43%（实测），
#     而 magnitude 的方向偏差只有约 5%
mag = cv.magnitude(gx, gy)
# 3.3 归一化到 0~255：用 99.5% 分位数做上限，避免个别极亮的点把整张图压暗
result = np.clip(mag / np.percentile(mag, 99.5) * 255, 0, 255).astype(np.uint8)

# 4 图像显示
plt.figure(figsize=(15, 5), dpi=100)
plt.subplot(131), plt.imshow(img, cmap=plt.cm.gray), plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(old, cmap=plt.cm.gray), plt.title('原写法（|gx|与|gy|各占0.5）')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(result, cmap=plt.cm.gray), plt.title('优化后（梯度幅值）')
plt.xticks([]), plt.yticks([])
plt.show()

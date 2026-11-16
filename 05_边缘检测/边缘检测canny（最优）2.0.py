import cv2 as cv
from matplotlib import pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1 图像读取
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg', 0)
if img is None:
    raise SystemExit('图片读取失败')

# 2 原写法：不去噪 + 固定阈值（低阈值 0 会把噪点也当成边缘）
canny_old = cv.Canny(img, 0, 100)

# 3 优化写法
# 3.1 先高斯去噪：Canny 对噪点很敏感
blur = cv.GaussianBlur(img, (5, 5), 1.4)
# 3.2 自适应阈值：Otsu 求出高阈值，低阈值取一半，换一张图也不用手调
#     （常见的"中位数法"在偏亮的图上会把高阈值顶到 255，细节全丢，所以这里用 Otsu）
high, _ = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
low = 0.5 * high
# 3.3 L2gradient=True：用精确的梯度幅值 sqrt(gx²+gy²)，默认的 |gx|+|gy| 对斜向边缘有偏差
canny_new = cv.Canny(blur, low, high, L2gradient=True)
print(f'自适应阈值：低={low:.0f}，高={high:.0f}')

# 4 图像展示
plt.figure(figsize=(15, 5), dpi=100)
plt.subplot(131), plt.imshow(img, cmap=plt.cm.gray), plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(canny_old, cmap=plt.cm.gray), plt.title('原写法（阈值0/100，不去噪）')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(canny_new, cmap=plt.cm.gray), plt.title('优化后（高斯去噪 + Otsu阈值）')
plt.xticks([]), plt.yticks([])
plt.show()

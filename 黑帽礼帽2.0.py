import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 解决中文方框（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1 读取图像：礼帽 / 黑帽是提取"比周围亮 / 暗的细节"，通常在灰度图上做
#   （彩色图会按 B、G、R 三个通道分别计算，颜色会失真）
img = cv.imread("/Users/pengxuezhi/Desktop/huiliyi.jpg", 0)
if img is None:
    raise SystemExit("图片读取失败")

# 2 创建核结构
kernel = np.ones((10, 10), np.uint8)

# 3 图像的礼帽和黑帽运算
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)  # 礼帽运算 = 原图 - 开运算（亮细节）
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)  # 黑帽运算 = 闭运算 - 原图（暗细节）

# 4 图像展示
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(tophat, cmap=plt.cm.gray)
axes[1].set_title("礼帽运算结果")
axes[2].imshow(blackhat, cmap=plt.cm.gray)
axes[2].set_title("黑帽运算结果")
plt.show()

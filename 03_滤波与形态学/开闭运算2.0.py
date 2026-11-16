import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 解决中文方框（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1 读取图像
img = cv.imread("/Users/pengxuezhi/Desktop/huiliyi.jpg")
if img is None:
    raise SystemExit("图片读取失败")

# 2 创建核结构
kernel = np.ones((10, 10), np.uint8)

# 3 图像的开闭运算
cvOpen = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)  # 开运算
cvClose = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)  # 闭运算

# 4 图像展示
# OpenCV 读出来是 BGR，matplotlib 按 RGB 显示：不用 [:,:,::-1] 转一下，红蓝会颠倒
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axes[0].imshow(img[:, :, ::-1])
axes[0].set_title("原图")
axes[1].imshow(cvOpen[:, :, ::-1])
axes[1].set_title("开运算结果")
axes[2].imshow(cvClose[:, :, ::-1])
axes[2].set_title("闭运算结果")
plt.show()

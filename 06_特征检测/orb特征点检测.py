import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 1 图像读取
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')

# 2 ORB 角点检测
# 实例化 ORB 对象，最多 500 个特征点
orb = cv.ORB_create(nfeatures=500)
# 检测关键点 + 计算描述子
kp, des = orb.detectAndCompute(img, None)

# 打印描述子维度
print(des.shape)

# 3 绘制关键点
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

# 4 展示
plt.figure(figsize=(10, 8), dpi=100)
plt.imshow(img2[:, :, ::-1])
plt.xticks([])
plt.yticks([])
plt.show()
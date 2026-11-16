import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 解决中文方框
plt.rcParams['font.sans-serif'] = ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 1 图像读取
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')
# 2 均值滤波
blur = cv.blur(img,(5,5))
# 3 图像显示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur[:,:,::-1]),plt.title('均值滤波后结果')
plt.xticks([]), plt.yticks([])
plt.show()
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 中文设置
plt.rcParams['font.sans-serif'] = ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 1 直接以灰度图的方式读入
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg',0)
# 2 统计灰度图
histr = cv.calcHist([img],[0],None,[256],[0,256])
# 3 绘制灰度图
plt.figure(figsize=(10,6),dpi=100)
plt.plot(histr)
plt.grid()
plt.show()
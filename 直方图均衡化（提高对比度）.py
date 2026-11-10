import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 中文设置
plt.rcParams['font.sans-serif'] = ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 1. 直接以灰度图的方式读入
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg',0)
# 2. 均衡化处理
dst = cv.equalizeHist(img)
# 3. 结果展示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img,cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(dst,cmap=plt.cm.gray)
axes[1].set_title("均衡化后结果")
plt.show()
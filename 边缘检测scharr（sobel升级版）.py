import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg',0)

# Scharr 算子，和Sobel用法几乎一样，ksize固定为-1代表Scharr
x = cv.Scharr(img, cv.CV_16S, 1, 0)
y = cv.Scharr(img, cv.CV_16S, 0, 1)

absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)
scharr_result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img,cmap=plt.cm.gray),plt.title('原图')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(scharr_result,cmap=plt.cm.gray),plt.title('Scharr检测')
plt.xticks([]),plt.yticks([])
plt.show()
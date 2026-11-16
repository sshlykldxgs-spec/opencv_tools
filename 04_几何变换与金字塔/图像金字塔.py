import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 图像读取
img = cv.imread("/Users/pengxuezhi/Desktop/huiliyi.jpg")

# 2 进行图像采样
up_img = cv.pyrUp(img)   # 上采样操作
img_1 = cv.pyrDown(img)  # 下采样操作
#img2 = cv.pyrDown(img_1) 循环次操作可以多次采样

# 3 图像显示
cv.imshow('enlarge', up_img)
cv.imshow('original', img)
cv.imshow('shrink', img_1)
cv.waitKey(0)
cv.destroyAllWindows()
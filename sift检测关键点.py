import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1 读取图像
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2 SIFT 关键点检测
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

# 3 在图像上绘制关键点（带大小、方向）
cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 4 图像显示
plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(img[:, :, ::-1])
plt.title('sift 检测')
plt.xticks([]), plt.yticks([])
plt.show()
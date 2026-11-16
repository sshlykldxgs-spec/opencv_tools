import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt

img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# LBP 参数：半径 1，采样点 8，均匀模式
radius = 1
n_points = 8 * radius
lbp_img = local_binary_pattern(gray, n_points, radius, method="uniform")

plt.figure(figsize=(10, 5), dpi=100)
plt.subplot(121)
plt.imshow(gray, cmap="gray")
plt.title("原图灰度")
plt.axis("off")

plt.subplot(122)
plt.imshow(lbp_img, cmap="gray")
plt.title("LBP 纹理图")
plt.axis("off")
plt.show()
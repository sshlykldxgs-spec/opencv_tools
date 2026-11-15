import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

hog = cv.HOGDescriptor()
# 计算 HOG 特征向量
feat = hog.compute(gray)
print("HOG 特征维度：", feat.shape)

# OpenCV 自带行人检测器版本（拓展）
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
boxes, _ = hog.detectMultiScale(img, winStride=(8, 8))
for x, y, w, h in boxes:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(img[:, :, ::-1])
plt.title("HOG 行人检测")
plt.axis("off")
plt.show()
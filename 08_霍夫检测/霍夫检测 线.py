import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 1.加载图片，转为二值图
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150)

# 2.霍夫直线变换
lines = cv.HoughLines(edges, 0.8, np.pi / 180, 150)

# 3.将检测的线绘制在图像上（注意是极坐标噢）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0))

# 4. 图像显示
plt.figure(figsize=(10, 8), dpi=100)
plt.imshow(img[:, :, ::-1])
plt.title('霍夫变换线检测')
plt.xticks([]), plt.yticks([])
plt.show()
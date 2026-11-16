import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img1 = cv.imread("/Users/pengxuezhi/Desktop/花.jpg")
img2 = cv.imread("/Users/pengxuezhi/Desktop/雨.jpg")

# 2 加法操作
img3 = cv.addWeighted(img1,0.7, img2,0.3,1)  # cv中的加法

# 3 图像显示
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,8), dpi=100)
axes[0].imshow(img3[:,:,::-1])
axes[0].set_title("cv中的加法")
plt.show()
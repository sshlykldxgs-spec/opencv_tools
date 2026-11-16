import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1 读取图像
img1 = cv.imread("/Users/pengxuezhi/Desktop/花.jpg")
img2 = cv.imread("/Users/pengxuezhi/Desktop/雨.jpg")
if img1 is None or img2 is None:
    raise SystemExit("图片读取失败")

# 两张图尺寸必须一样才能相加，不一样就把 img2 缩放到和 img1 同样大小（否则会直接报错）
if img1.shape != img2.shape:
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv.INTER_LINEAR)

# 2 加法操作
img3 = cv.add(img1, img2)  # cv中的加法：超过255就饱和成255（越加越亮）
img4 = img1 + img2  # 直接相加：numpy 对256取余，超过255会回绕成很小的数（出现怪异的暗斑）

# 3 图像显示
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(img3[:, :, ::-1])
axes[0].set_title("cv中的加法")
axes[1].imshow(img4[:, :, ::-1])
axes[1].set_title("直接相加")
plt.show()

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

# 两张图尺寸必须一样才能混合，不一样就把 img2 缩放到和 img1 同样大小（否则会直接报错）
if img1.shape != img2.shape:
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv.INTER_LINEAR)

# 2 加权混合：dst = img1*0.7 + img2*0.3 + 1
img3 = cv.addWeighted(img1, 0.7, img2, 0.3, 1)  # cv中的加法

# 3 图像显示
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=100)
axes[0].imshow(img1[:, :, ::-1])
axes[0].set_title("图1（权重0.7）")
axes[1].imshow(img2[:, :, ::-1])
axes[1].set_title("图2（权重0.3）")
axes[2].imshow(img3[:, :, ::-1])
axes[2].set_title("cv中的加权混合")
plt.show()

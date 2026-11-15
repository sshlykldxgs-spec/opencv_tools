import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 1 读取图像
img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')

# 2 FAST 角点检测
# 创建 Fast 对象，阈值 30
fast = cv.FastFeatureDetector_create(threshold=30)

# 检测关键点
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

# 打印参数
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression: {}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

# 关闭非极大值抑制
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
img3 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

# 双图对比展示
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(img2[:, :, ::-1])
axes[0].set_title("加入非极大值抑制")
axes[0].axis("off")

axes[1].imshow(img3[:, :, ::-1])
axes[1].set_title("未加入非极大值抑制")
axes[1].axis("off")

plt.show()
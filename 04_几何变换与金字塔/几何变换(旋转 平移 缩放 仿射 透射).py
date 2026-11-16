#缩放
# 插值	含义
# cv2.INTER_LINEAR	双线性插值法
# cv2.INTER_NEAREST	最近邻插值
# cv2.INTER_AREA	像素区域重采样(默认)
# cv2.INTER_CUBIC	双三次插值
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取图片
img1 = cv.imread("/Users/pengxuezhi/Desktop/花.jpg")

# 2.图像缩放
# 2.1 绝对尺寸
rows,cols = img1.shape[:2]
res = cv.resize(img1,(2*cols,2*rows),interpolation=cv.INTER_CUBIC)

# 2.2 相对尺寸
res1 = cv.resize(img1,None,fx=0.5,fy=0.5)

# 3 图像显示
# 3.1 使用opencv显示图像(不推荐)
cv.imshow("original",img1)
cv.imshow("enlarge",res)
cv.imshow("shrink",res1)
cv.waitKey(0)

# 3.2 使用matplotlib显示图像
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,8),dpi=100)
axes[0].imshow(res[:,:,::-1])
axes[0].set_title("绝对尺度（放大）")
axes[1].imshow(img1[:,:,::-1])
axes[1].set_title("原图")
axes[2].imshow(res1[:,:,::-1])
axes[2].set_title("相对尺度（缩小）")
plt.show()

#图像平移
M = np.float([[1,0,100],[0,1,50]])
res2 = cv.warpPerspective(img1,M,(cols,rows))
plt.imshow(res2[:,:,::-1])

#图像旋转
# 1 读取图像
img = cv.imread("/Users/pengxuezhi/Desktop/huiliyi.jpg")

# 2 图像旋转
rows,cols = img.shape[:2]
# 2.1 生成旋转矩阵
M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
# 2.2 进行旋转变换
dst = cv.warpAffine(img,M,(cols,rows))

# 3 图像展示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("原图")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("旋转后结果")
plt.show()

#4 仿射变换
img = cv.imread("/Users/pengxuezhi/Desktop/花.jpg")

rows,cols = img.shape[:2]
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[100,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("原图")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("仿射后结果")
plt.show()

#透射变换
img = cv.imread("/Users/pengxuezhi/Desktop/花.jpg")

rows,cols = img.shape[:2]
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[100,145],[300,100],[80,290],[310,300]])
T = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,T,(cols,rows))

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("原图")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("透视后结果")
plt.show()
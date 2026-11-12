#缩放
# 插值	含义
# cv2.INTER_LINEAR	双线性插值法(默认)
# cv2.INTER_NEAREST	最近邻插值
# cv2.INTER_AREA	像素区域重采样(缩小图像时首选，不容易出现锯齿)
# cv2.INTER_CUBIC	双三次插值(放大图像时常用，更平滑)
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取图片
img1 = cv.imread("/Users/pengxuezhi/Desktop/花.jpg")
if img1 is None:
    raise SystemExit("图片读取失败")

# 2.图像缩放
# 2.1 绝对尺寸（放大用 INTER_CUBIC）
rows,cols = img1.shape[:2]
res = cv.resize(img1,(2*cols,2*rows),interpolation=cv.INTER_CUBIC)

# 2.2 相对尺寸（缩小用 INTER_AREA）
res1 = cv.resize(img1,None,fx=0.5,fy=0.5,interpolation=cv.INTER_AREA)

# 3 图像显示
# 3.1 使用opencv显示图像(不推荐)
cv.imshow("original",img1)
cv.imshow("enlarge",res)
cv.imshow("shrink",res1)
cv.waitKey(0)
cv.destroyAllWindows()  # 按任意键后关掉窗口，再继续往下

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
# 平移矩阵是 2x3，要用 warpAffine（warpPerspective 需要 3x3 矩阵）；np.float 不能这样用，要写 np.float32
M = np.float32([[1,0,100],[0,1,50]])  # 向右 100，向下 50
res2 = cv.warpAffine(img1,M,(cols,rows))
plt.imshow(res2[:,:,::-1])
plt.title("平移结果")
plt.show()

#图像旋转
# 1 读取图像
img = cv.imread("/Users/pengxuezhi/Desktop/huiliyi.jpg")
if img is None:
    raise SystemExit("图片读取失败")

# 2 图像旋转
rows,cols = img.shape[:2]
# 2.1 生成旋转矩阵
M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
# 2.2 计算旋转后完整图像需要的画布大小，并把中心平移过去（原写法画布大小不变，旋转后四角会被裁掉）
cos, sin = abs(M[0,0]), abs(M[0,1])
new_w = int(round(rows*sin + cols*cos))
new_h = int(round(rows*cos + cols*sin))
M[0,2] += new_w/2 - cols/2
M[1,2] += new_h/2 - rows/2
# 2.3 进行旋转变换（旋转会产生小数坐标，用 INTER_CUBIC 比默认的双线性更清晰）
dst = cv.warpAffine(img,M,(new_w,new_h),flags=cv.INTER_CUBIC)

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

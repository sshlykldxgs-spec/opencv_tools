import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = np.zeros((256,226,3),np.uint8)
plt.axis('off')
plt.show()
plt.imshow(img[:,:,::-1])
# 获取某个像素点的值
px = img[100,100,1]
# 仅获取蓝色通道的强度值
blue = img[100,100,0]
# 修改某个位置的像素值
img[100,100] = [255,255,255]
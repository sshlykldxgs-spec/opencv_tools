import cv2 as cv
import  numpy as np
import matplotlib.pyplot as plt

huiliyi = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')
# plt.imshow(huiliyi[:,:,::-1])
# plt.show()
# b,g,r = cv.split(huiliyi)
# plt.imshow(b,cmap=plt.cm.grey)
# plt.show()
# img2 = cv.merge(b,g,r)
# plt.imshow(img2[:,:,::-1])
# plt.show()
# grey = cv.cvtColor(huiliyi, cv.COLOR_BGR2GRAY)
# plt.imshow(grey,cmap=plt.cm.gray)
# plt.show()
hsv = cv.cvtColor(huiliyi, cv.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.show()
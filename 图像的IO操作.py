import cv2 as cv
import  numpy as np
import matplotlib.pyplot as plt

img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg',0)

if img is None:
    print("图片读取失败")
else:
    cv.imshow("image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("messigray.png", img)







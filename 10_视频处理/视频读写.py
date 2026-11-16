import numpy as np
import cv2 as cv

# 打开视频文件，也可以填 0 调用摄像头
cap = cv.VideoCapture(0)

# 判断是否打开成功
while cap.isOpened():
    ret, frame = cap.read()
    # ret：是否读到帧；frame：当前画面
    if ret == True:
        cv.imshow('frame', frame)
        # 等待 25ms，按 q 退出
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源，销毁窗口（必写！）
cap.release()
cv.destroyAllWindows()
import numpy as np
import cv2 as cv

# 打开视频
cap = cv.VideoCapture(0)

# 读第一帧，手动框选目标位置
ret, frame = cap.read()
#起始框
r, h, c, w = 197, 141, 0, 208
track_window = (c, r, w, h)

# 截取 ROI
roi = frame[r:r + h, c:c + w]
# 转 HSV
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# 过滤低亮度
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
# 计算 H 通道直方图
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# 归一化
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# 迭代终止条件：最大迭代 10，窗口漂移小于 1
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 直方图反向投影
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # MeanShift 跟踪
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # 画框
        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv.imshow('frame', img2)

        # 按 q 退出
        if cv.waitKey(60) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
cv.destroyAllWindows()
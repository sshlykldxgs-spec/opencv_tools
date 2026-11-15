import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("打开摄像头失败")
    exit()

# 先读一帧用来框选
ret, frame = cap.read()
# 鼠标拖框，回车确认
bbox = cv.selectROI("框选目标", frame, fromCenter=False, showCrosshair=True)
x, y, w, h = bbox
track_window = (x, y, w, h)

# 取出被框的区域
roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# 过滤暗像素
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# 迭代终止条件
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 反向投影：亮区 = 和目标颜色匹配
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 核心跟踪
    ret, track_window = cv.meanShift(dst, track_window, term_crit)
    xx, yy, ww, hh = track_window

    # 画框
    out = cv.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
    cv.imshow("跟踪画面", out)
    cv.imshow("反向投影", dst)   # 重点！看这个图就懂了

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
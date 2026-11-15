import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
ret, frame = cap.read()

# 手动画框
bbox = cv.selectROI("框选目标", frame, fromCenter=False, showCrosshair=True)
x, y, w, h = bbox
track_window = (x, y, w, h)

roi = frame[y:y + h, x:x + w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv.CamShift(dst, track_window, term_crit)
    pts = cv.boxPoints(ret)
    pts = pts.astype(np.int32)
    frame = cv.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv.imshow("CamShift 跟踪", frame)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
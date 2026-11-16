import cv2 as cv
import numpy as np

# 1. 读取视频
cap = cv.VideoCapture(0)

# 增加判断（非常重要）
if not cap.isOpened():
    print("打开视频失败！检查文件路径")
    exit()

# 2. 获取宽高
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# 3. 构造写入器
# MJPG → 保存 avi（讲义原版）
fourcc = cv.VideoWriter_fourcc('M','J','P','G')
out = cv.VideoWriter('outpy.avi', fourcc, 10, (frame_width, frame_height))

# ========== Mac 更稳的备选（推荐替换上面两行） ==========
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('outpy.mp4', fourcc, 15, (frame_width, frame_height))

# 4. 循环读帧、写帧
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv.imshow("frame", frame)
        # 按 q 退出
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 5. 释放资源（必写！）
cap.release()
out.release()
cv.destroyAllWindows()
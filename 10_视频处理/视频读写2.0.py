import sys
import time
import cv2 as cv

# ====================== 可调参数 ======================
SOURCE = 0        # 摄像头编号；也可以写视频文件路径，例如 'slow.flv'
LOOP = False      # 播放视频文件时，True = 播完自动从头再播


def play(cap, fps, is_camera, show=True, now=time.perf_counter):
    """播放/预览。返回显示的帧数"""
    # 视频文件：按原帧率播放，并扣掉读帧、显示本身花掉的时间（原写法固定 waitKey(25)，不管视频是 24、30 还是 60 帧）
    # 摄像头：read() 本身会等到下一帧，所以 waitKey(1) 就够了（waitKey(25) 会让每帧多等 25ms，实际帧率掉到十几帧）
    period = 1.0 / fps if (not is_camera and fps > 1) else 0.0
    count = 0
    while True:
        t0 = now()
        ret, frame = cap.read()
        if not ret:
            if LOOP and not is_camera and count > 0:
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        count += 1
        if show:
            cv.imshow('frame', frame)
            wait = max(1, int((period - (now() - t0)) * 1000)) if period else 1
            if cv.waitKey(wait) & 0xFF in (ord('q'), 27):  # q 或 ESC 退出
                break
    return count


if __name__ == '__main__':
    cap = cv.VideoCapture(SOURCE)
    # 原写法 `while cap.isOpened()` 在打开失败时什么提示都没有就直接结束了
    if not cap.isOpened():
        sys.exit('打开视频失败！摄像头被占用/没有权限，或者检查文件路径：' + str(SOURCE))
    is_camera = isinstance(SOURCE, int)
    n = play(cap, cap.get(cv.CAP_PROP_FPS), is_camera)
    print('共显示', n, '帧')

    # 释放资源，销毁窗口（必写！）
    cap.release()
    cv.destroyAllWindows()

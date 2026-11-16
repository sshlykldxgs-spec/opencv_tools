import sys
import numpy as np
import cv2 as cv

# ====================== 可调参数 ======================
SOURCE = 0                       # 摄像头编号；也可以写视频文件路径，例如 'slow.flv'
INIT_BOX = None                # 起始框 (x, y, w, h)；None = 在第一帧上用鼠标框选
USE_CAMSHIFT = False           # True: CamShift（框会随目标缩放/旋转）；False: meanShift（框大小固定）
SHOW_BACKPROJECTION = True     # 是否另开窗口显示反向投影（亮 = 和目标颜色匹配）

# ===================== 颜色追踪核心（4 个追踪脚本共用同一份） =====================
# 有效像素：饱和度 >= 60、亮度 >= 32。太灰、太暗的像素色相不稳定，不参与建模，也不参与反向投影
HSV_LO = np.array((0, 60, 32), np.uint8)
HSV_HI = np.array((180, 255, 255), np.uint8)
TERM_CRIT = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
LOST_CONF = 0.35      # 窗口中心区域的颜色匹配度低于"初始框中心区域的匹配度"的这个比例，就认为目标丢了
RELOCK_CONF = 0.5     # 重找时，最佳位置的匹配度要达到初始框的这个比例，才认为找回来了
LOST_FRAMES = 3       # 连续这么多帧匹配度过低，才开始重找（避免偶尔一帧抖动就乱跳）
RELOCK_GROW = 0.25    # 重找范围：从"1 倍框大小"开始，每多丢一帧扩大这么多倍
RELOCK_MAX = 3.0      # 重找范围最大不超过"3 倍框大小"（只在丢失位置附近找，不去全图找，免得被远处的同色物体勾走）


class ColorTracker:
    def __init__(self, frame, box, use_camshift=True):
        H, W = frame.shape[:2]
        x, y, w, h = [int(round(v)) for v in box]
        x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        roi = hsv[y:y + h, x:x + w]
        mask = cv.inRange(roi, HSV_LO, HSV_HI)
        if cv.countNonZero(mask) < 0.1 * w * h:
            raise ValueError('框选区域里颜色太灰或太暗，没法按颜色追踪，请换一个颜色鲜明的目标')
        # H-S 二维直方图：只用色相 H 的话，同色相但饱和度不同的东西（比如浅橙色的墙）也会被当成目标
        self.hist = cv.calcHist([roi], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv.normalize(self.hist, self.hist, 0, 255, cv.NORM_MINMAX)
        self.use_camshift = use_camshift
        self.window = (x, y, w, h)
        self.size = (w, h)
        dst = self._backproject(hsv)
        self.ref = max(self._inner_mean(dst, self.window), 1e-3)  # 初始框中心区域的平均匹配度，作为"正常追踪"的基准
        self.lost = 0
        self.conf = 1.0
        self.last_backprojection = dst

    @staticmethod
    def _inner_mean(dst, win):
        """窗口中心一半区域内的平均匹配度。只看中心：meanShift 的窗口大小不会变，目标缩小时窗口边缘会框进背景，
        拿整个窗口求平均会把"没丢"误判成"丢了"""
        x, y, w, h = win
        ix, iy = max(0, x + w // 4), max(0, y + h // 4)
        patch = dst[iy:iy + max(1, h // 2), ix:ix + max(1, w // 2)]
        return float(patch.mean()) if patch.size else 0.0

    def _backproject(self, hsv):
        dst = cv.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        return cv.bitwise_and(dst, cv.inRange(hsv, HSV_LO, HSV_HI))  # 屏蔽灰/暗像素：原写法这些像素会产生杂乱的响应

    def update(self, frame):
        """返回 (窗口 (x,y,w,h), 旋转框 4x2 或 None, 是否丢失)。丢失时窗口停在最后可信的位置"""
        H, W = frame.shape[:2]
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = self._backproject(hsv)
        self.last_backprojection = dst

        if self.use_camshift:
            rot, win = cv.CamShift(dst, self.window, TERM_CRIT)
        else:
            _, win = cv.meanShift(dst, self.window, TERM_CRIT)
            rot = None
        x, y, w, h = [int(v) for v in win]
        conf = self._inner_mean(dst, (x, y, w, h)) / self.ref
        self.conf = conf

        if conf >= LOST_CONF:
            self.lost = 0
            if self.use_camshift:
                # CamShift 的窗口偶尔会缩成一小点或胀得很大，限制在初始大小的 0.4~2.5 倍
                w0, h0 = self.size
                cx, cy = x + w / 2, y + h / 2
                w, h = int(np.clip(w, 0.4 * w0, 2.5 * w0)), int(np.clip(h, 0.4 * h0, 2.5 * h0))
                x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
            self.window = (x, y, w, h)
            return self.window, (cv.boxPoints(rot) if rot is not None else None), False

        self.lost += 1
        if self.lost >= LOST_FRAMES:
            # 目标丢了（被挡住、短暂离开）：在丢失位置附近找匹配度最高的地方，范围随丢失时间慢慢扩大，找到了就重新锁定
            w0, h0 = self.size
            m = max(w0, h0)
            R = int(min(m * (1.0 + RELOCK_GROW * self.lost), m * RELOCK_MAX))
            wx, wy, ww, wh = self.window
            cx, cy = wx + ww // 2, wy + wh // 2
            rx0, ry0 = max(0, cx - R), max(0, cy - R)
            rx1, ry1 = min(W, cx + R), min(H, cy + R)
            score = cv.blur(dst, (max(1, w0 // 2), max(1, h0 // 2)))[ry0:ry1, rx0:rx1]
            if score.size:
                _, mx, _, loc = cv.minMaxLoc(score)
                if mx / self.ref >= RELOCK_CONF:
                    x0 = int(np.clip(rx0 + loc[0] - w0 // 2, 0, max(0, W - w0)))
                    y0 = int(np.clip(ry0 + loc[1] - h0 // 2, 0, max(0, H - h0)))
                    self.window = (x0, y0, w0, h0)
                    self.lost = 0
                    return self.window, None, False
        return self.window, None, True


def open_source():
    cap = cv.VideoCapture(SOURCE)
    if not cap.isOpened():
        sys.exit('打开视频源失败：' + str(SOURCE))
    return cap


def first_frame(cap):
    # 摄像头刚打开时前几帧经常是全黑或曝光没调好的，先丢掉几帧再取（原写法直接用第 1 帧建模型）
    if isinstance(SOURCE, int):
        for _ in range(10):
            cap.read()
    ok, frame = cap.read()
    if not ok:
        sys.exit('读取第一帧失败')
    return frame


def select_box(frame):
    win = '框选目标（鼠标拖框，回车确认）'
    x, y, w, h = cv.selectROI(win, frame, fromCenter=False, showCrosshair=True)
    cv.destroyWindow(win)
    if w == 0 or h == 0:  # 原写法这里没有检查，直接按 ESC / 取消会得到空框然后报错
        sys.exit('没有框选目标')
    return (x, y, w, h)


def draw(frame, window, poly, lost):
    if lost:  # 目标被挡住或离开：框停在最后可信的位置并标红，等目标回来自动找回
        x, y, w, h = window
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, 'LOST', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif poly is not None:
        cv.polylines(frame, [np.int32(poly)], True, (0, 255, 0), 2)
    else:
        x, y, w, h = window
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def main():
    cap = open_source()
    frame = first_frame(cap)

    tracker = None
    if INIT_BOX is not None:
        try:
            tracker = ColorTracker(frame, INIT_BOX, USE_CAMSHIFT)
        except ValueError as e:
            print('起始框不可用：', e, '→ 改为手动框选')
    while tracker is None:
        try:
            tracker = ColorTracker(frame, select_box(frame), USE_CAMSHIFT)
        except ValueError as e:
            print(e)

    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps) if (not isinstance(SOURCE, int) and fps > 1) else 1  # 视频文件按原帧率播放
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        window, poly, lost = tracker.update(frame)
        cv.imshow('tracking', draw(frame, window, poly, lost))
        if SHOW_BACKPROJECTION:
            cv.imshow('backprojection', tracker.last_backprojection)
        if cv.waitKey(delay) & 0xFF in (ord('q'), 27):  # q 或 ESC 退出
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

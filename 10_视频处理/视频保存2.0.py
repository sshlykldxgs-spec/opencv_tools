import sys
import time
import cv2 as cv

# ====================== 可调参数 ======================
SOURCE = 0              # 摄像头编号；也可以写视频文件路径（此时是"转存/重新编码"）
OUT_PATH = 'outpy.mp4'  # 输出文件；.mp4 用 mp4v 编码，其它后缀（如 .avi）用 MJPG
DEFAULT_FPS = 30        # 摄像头读不出帧率时用这个


class FrameClock:
    """按真实时间决定"每个摄像头画面要写几次"，保证保存下来的视频时长 = 实际录制时长。
    原写法把帧率写死成 10：摄像头实际约 30 帧/秒，存出来的视频就会慢放 3 倍；
    即使把帧率改对，imshow + waitKey(25) 也会让循环跑不满摄像头的帧率，视频照样会比实际快或慢。"""

    def __init__(self, fps):
        self.fps = fps
        self.t0 = None
        self.written = 0

    def due(self, now):
        """现在这一帧应该写几次：来得慢就重复写，来得快就跳过"""
        if self.t0 is None:
            self.t0 = now
        n = max(0, int((now - self.t0) * self.fps) + 1 - self.written)
        self.written += n
        return n


def open_source():
    cap = cv.VideoCapture(SOURCE)
    if not cap.isOpened():
        sys.exit('打开视频失败！摄像头被占用/没有权限，或者检查文件路径：' + str(SOURCE))
    return cap


def make_writer(path, fps, size):
    """mp4v + .mp4；打不开就退回 MJPG + .avi（几乎所有环境都能写）。size 是 (宽, 高)"""
    if path.lower().endswith('.mp4'):
        out = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'mp4v'), fps, size)
        if out.isOpened():
            return out, path
        path = path[:-4] + '.avi'
        print('mp4v 编码器不可用，改存 MJPG 的 avi：', path)
    out = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'MJPG'), fps, size)
    if not out.isOpened():
        sys.exit('创建输出文件失败：' + path)
    return out, path


def record(cap, out, fps, realtime, now=time.perf_counter, show=True):
    """循环读帧、写帧，返回写入的帧数。realtime=True（摄像头）按真实时间补帧/丢帧；False（视频文件）逐帧原样写"""
    clock = FrameClock(fps)
    size = (int(out_size[0]), int(out_size[1]))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame.shape[1], frame.shape[0]) != size:  # 帧尺寸和写入器不一致时 VideoWriter 会悄悄丢帧，导致视频损坏
            frame = cv.resize(frame, size)
        n = clock.due(now()) if realtime else 1
        for _ in range(n):
            out.write(frame)
        count += n
        if show:
            cv.imshow('frame', frame)
            # 录制时 waitKey 只需要 1ms：waitKey(25) 会让每帧白白多等 25ms
            if cv.waitKey(1) & 0xFF in (ord('q'), 27):  # q 或 ESC 退出
                break
    return count


if __name__ == '__main__':
    cap = open_source()
    is_camera = isinstance(SOURCE, int)

    # 帧率：摄像头用它报告的帧率（读不出来就用默认值）；视频文件用文件自己的帧率
    fps = cap.get(cv.CAP_PROP_FPS)
    if not (1 <= fps <= 240):
        fps = DEFAULT_FPS

    # 画面尺寸以真正读到的第一帧为准（cap.get 报的宽高和实际帧不一致时，视频会损坏）
    ret, first = cap.read()
    if not ret:
        sys.exit('读取第一帧失败')
    out_size = (first.shape[1], first.shape[0])
    out, path = make_writer(OUT_PATH, fps, out_size)

    # 把第一帧放回去：用一个小包装，让 record 先吐出 first 再继续读
    class _Replay:
        def __init__(self, cap, first):
            self.cap, self.first = cap, first

        def read(self):
            if self.first is not None:
                f, self.first = self.first, None
                return True, f
            return self.cap.read()

    try:
        n = record(_Replay(cap, first), out, fps, realtime=is_camera)
    finally:  # 即使中途出错/按 Ctrl+C，也要释放，否则视频文件打不开
        cap.release()
        out.release()
        cv.destroyAllWindows()
    print(f'已保存 {path}：{n} 帧，{fps:.1f} 帧/秒，时长约 {n / fps:.1f} 秒')

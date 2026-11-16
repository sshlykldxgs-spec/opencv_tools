import os
import sys
import cv2 as cv
import numpy as np

# =========================
# 1. 分类器：先找脚本旁边的 haarcascades，找不到再用 OpenCV 自带的
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_DIRS = [os.path.join(BASE_DIR, "haarcascades"), cv.data.haarcascades]


def load_cascade(filename, required=True):
    for d in HAAR_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            c = cv.CascadeClassifier(path)
            if not c.empty():
                return c
    if required:
        raise FileNotFoundError(f"找不到分类器：{filename}（找过：{HAAR_DIRS}）")
    return None


def load_models():
    return dict(
        face=load_cascade("haarcascade_frontalface_default.xml"),
        profile=load_cascade("haarcascade_profileface.xml"),
        eye=load_cascade("haarcascade_eye.xml"),
        smile=load_cascade("haarcascade_smile.xml"),
        nose=load_cascade("haarcascade_mcs_nose.xml", required=False),
        upper=load_cascade("haarcascade_upperbody.xml", required=False),
        full=load_cascade("haarcascade_fullbody.xml", required=False),
    )


def iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def detect(frame, models, det_width=640, with_body=True):
    """对一帧做检测，返回 dict：face / profile / eye / smile / nose / upper / full，每项是 [(x, y, w, h), ...]（原图坐标）。
    相比原写法：
      1 检测在缩小到宽度 det_width 的灰度图上做（1280x720 的摄像头画面少算 3/4 的像素），框再放大回原图
      2 侧脸：原图和左右镜像各检测一次（OpenCV 的侧脸模型只认"脸朝一个方向"，原写法脸朝另一边就检测不到）
      3 侧脸与正脸的去重用 IoU（原写法只比较左上角坐标，大小不同的重叠框会漏掉）
      4 眼睛只在人脸上半部分找（原写法在整张脸里找，嘴、鼻孔、眉毛都可能被当成眼睛）
      5 没有 nose / upper / full 模型时自动跳过"""
    h, w = frame.shape[:2]
    s = min(1.0, det_width / w)
    small = cv.resize(frame, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else frame
    gray = cv.equalizeHist(cv.cvtColor(small, cv.COLOR_BGR2GRAY))
    gh, gw = gray.shape
    inv = lambda r: tuple(int(round(v / s)) for v in r)  # 缩小图坐标 → 原图坐标
    res = {k: [] for k in ("face", "profile", "eye", "smile", "nose", "upper", "full")}

    min_face = max(20, int(60 * s))
    faces = models["face"].detectMultiScale(gray, 1.2, 5, minSize=(min_face, min_face))
    for (x, y, fw, fh) in faces:
        res["face"].append(inv((x, y, fw, fh)))
        roi = gray[y:y + fh, x:x + fw]
        top = roi[:int(fh * 0.6), :]  # 眼睛只会在脸的上 60%
        for (ex, ey, ew, eh) in models["eye"].detectMultiScale(top, 1.1, 6, minSize=(max(8, int(fw * 0.1)),) * 2):
            res["eye"].append(inv((x + ex, y + ey, ew, eh)))
        mouth = roi[int(fh * 0.5):, :]
        for (sx, sy, sw, sh) in models["smile"].detectMultiScale(mouth, 1.7, 22, minSize=(max(12, int(fw * 0.25)), 10)):
            res["smile"].append(inv((x + sx, y + int(fh * 0.5) + sy, sw, sh)))
        if models["nose"] is not None:
            nx0, ny0 = int(fw * 0.2), int(fh * 0.25)
            nroi = roi[ny0:int(fh * 0.8), nx0:int(fw * 0.8)]
            for (nx, ny, nw, nh) in models["nose"].detectMultiScale(nroi, 1.2, 5, minSize=(12, 12)):
                res["nose"].append(inv((x + nx0 + nx, y + ny0 + ny, nw, nh)))

    # 侧脸：原图 + 镜像
    cand = [tuple(r) for r in models["profile"].detectMultiScale(gray, 1.2, 5, minSize=(min_face, min_face))]
    for (x, y, pw, ph) in models["profile"].detectMultiScale(cv.flip(gray, 1), 1.2, 5, minSize=(min_face, min_face)):
        cand.append((gw - x - pw, y, pw, ph))
    kept = []
    for r in cand:
        if any(iou(r, f) > 0.3 for f in faces) or any(iou(r, k) > 0.3 for k in kept):
            continue
        kept.append(r)
    res["profile"] = [inv(r) for r in kept]

    if with_body:
        if models["upper"] is not None:
            for r in models["upper"].detectMultiScale(gray, 1.2, 5, minSize=(max(40, int(80 * s)), max(50, int(100 * s)))):
                res["upper"].append(inv(r))
        if models["full"] is not None:
            for r in models["full"].detectMultiScale(gray, 1.2, 5, minSize=(max(40, int(80 * s)), max(75, int(150 * s)))):
                res["full"].append(inv(r))
    return res


COLORS = dict(face=(0, 255, 0), profile=(0, 165, 255), eye=(255, 0, 0), smile=(0, 255, 255),
              nose=(255, 0, 255), upper=(255, 255, 0), full=(255, 0, 255))
LABELS = dict(face="Face", profile="Profile", eye="Eye", smile="Smile", nose="Nose", upper="Upper Body", full="Full Body")


def draw(frame, res):
    for k, rects in res.items():
        for (x, y, w, h) in rects:
            cv.rectangle(frame, (x, y), (x + w, y + h), COLORS[k], 2)
            cv.putText(frame, LABELS[k], (x, max(12, y - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[k], 1)
    cv.putText(frame, f"Faces: {len(res['face']) + len(res['profile'])}", (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def open_camera(index=0):
    """macOS 优先用 AVFoundation，打不开或不是 mac 就退回默认后端"""
    backends = [cv.CAP_AVFOUNDATION] if sys.platform == "darwin" else []
    backends.append(cv.CAP_ANY)
    for b in backends:
        cap = cv.VideoCapture(index, b)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None


if __name__ == "__main__":
    models = load_models()
    cap = open_camera(0)
    if cap is None:
        raise SystemExit("无法打开摄像头，请检查摄像头权限（macOS：系统设置 → 隐私与安全性 → 摄像头）")
    print("摄像头启动成功！按 Q 或 ESC 退出")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            cv.imshow("Real-time Detection", draw(frame, detect(frame, models)))
            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
        print("摄像头已关闭")

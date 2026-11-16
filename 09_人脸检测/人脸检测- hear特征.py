
import cv2 as cv
import os

# =========================
# 1. 路径配置
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_DIR = os.path.join(BASE_DIR, "haarcascades")


def load_cascade(filename, required=True):
    path = os.path.join(HAAR_DIR, filename)

    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"找不到分类器：{path}")
        print(f"可选分类器不存在，跳过：{filename}")
        return None

    classifier = cv.CascadeClassifier(path)

    if classifier.empty():
        if required:
            raise RuntimeError(f"分类器加载失败：{filename}")
        return None

    print(f"加载成功：{filename}")
    return classifier


# =========================
# 2. 加载检测模型
# =========================

face_cas = load_cascade(
    "haarcascade_frontalface_default.xml"
)

profile_cas = load_cascade(
    "haarcascade_profileface.xml"
)

eye_cas = load_cascade(
    "haarcascade_eye.xml"
)

smile_cas = load_cascade(
    "haarcascade_smile.xml"
)

nose_cas = load_cascade(
    "haarcascade_mcs_nose.xml",
    required=False
)

upperbody_cas = load_cascade(
    "haarcascade_upperbody.xml",
    required=False
)

fullbody_cas = load_cascade(
    "haarcascade_fullbody.xml",
    required=False
)


# =========================
# 3. 打开 Mac 摄像头
# =========================

cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError(
        "无法打开摄像头，请检查 macOS 摄像头权限"
    )

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

print("摄像头启动成功！")
print("按 Q 或 ESC 退出")


# =========================
# 4. 实时检测
# =========================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        # -------------------------
        # 人脸检测
        # -------------------------

        faces = face_cas.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        face_regions = []

        for (x, y, w, h) in faces:

            face_regions.append((x, y, w, h))

            cv.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv.putText(
                frame,
                "Face",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # 截取人脸区域
            roi_gray = gray[y:y+h, x:x+w]

            # -------------------------
            # 眼睛检测
            # -------------------------

            eyes = eye_cas.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(12, 12)
            )

            for (ex, ey, ew, eh) in eyes:

                cv.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (255, 0, 0),
                    2
                )

                cv.putText(
                    frame,
                    "Eye",
                    (x + ex, y + ey - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1
                )

            # -------------------------
            # 微笑检测
            # -------------------------

            smile_roi = roi_gray[int(h * 0.5):h, :]

            smiles = smile_cas.detectMultiScale(
                smile_roi,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 15)
            )

            for (sx, sy, sw, sh) in smiles:

                cv.rectangle(
                    frame,
                    (x + sx, y + int(h * 0.5) + sy),
                    (x + sx + sw, y + int(h * 0.5) + sy + sh),
                    (0, 255, 255),
                    2
                )

                cv.putText(
                    frame,
                    "Smile",
                    (x + sx, y + int(h * 0.5) + sy - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

            # -------------------------
            # 鼻子检测
            # -------------------------

            if nose_cas is not None:

                nose_roi = roi_gray[
                    int(h * 0.25):int(h * 0.8),
                    int(w * 0.2):int(w * 0.8)
                ]

                noses = nose_cas.detectMultiScale(
                    nose_roi,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(15, 15)
                )

                for (nx, ny, nw, nh) in noses:

                    cv.rectangle(
                        frame,
                        (
                            x + int(w * 0.2) + nx,
                            y + int(h * 0.25) + ny
                        ),
                        (
                            x + int(w * 0.2) + nx + nw,
                            y + int(h * 0.25) + ny + nh
                        ),
                        (255, 0, 255),
                        2
                    )

                    cv.putText(
                        frame,
                        "Nose",
                        (
                            x + int(w * 0.2) + nx,
                            y + int(h * 0.25) + ny - 5
                        ),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1
                    )

        # -------------------------
        # 侧脸检测
        # -------------------------

        profiles = profile_cas.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in profiles:

            # 避免与已经检测到的正脸重复标记
            if any(
                abs(x - fx) < fw * 0.5
                and abs(y - fy) < fh * 0.5
                for fx, fy, fw, fh in face_regions
            ):
                continue

            cv.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 165, 255),
                2
            )

            cv.putText(
                frame,
                "Profile Face",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

        # -------------------------
        # 上半身检测
        # -------------------------

        if upperbody_cas is not None:

            bodies = upperbody_cas.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 100)
            )

            for (x, y, w, h) in bodies:

                cv.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    2
                )

                cv.putText(
                    frame,
                    "Upper Body",
                    (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        # -------------------------
        # 全身检测
        # -------------------------

        if fullbody_cas is not None:

            bodies = fullbody_cas.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 150)
            )

            for (x, y, w, h) in bodies:

                cv.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 255),
                    2
                )

                cv.putText(
                    frame,
                    "Full Body",
                    (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2
                )

        # -------------------------
        # 显示检测状态
        # -------------------------

        cv.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv.imshow("Mac Real-time Detection", frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv.destroyAllWindows()
    print("摄像头已关闭")

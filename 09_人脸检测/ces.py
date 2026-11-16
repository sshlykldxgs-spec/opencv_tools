import cv2 as cv
import os

# 获取 OpenCV 自带的 Haar 文件目录
haar_path = cv.data.haarcascades

# 定义人脸分类器路径
face_path = os.path.join(
    haar_path,
    "haarcascade_frontalface_default.xml"
)

# 定义眼睛分类器路径
eye_path = os.path.join(
    haar_path,
    "haarcascade_eye.xml"
)

# 检查文件是否存在
print("OpenCV版本：", cv.__version__)
print("OpenCV目录：", cv.__file__)
print("Haar目录：", haar_path)
print("人脸文件路径：", face_path)
print("人脸文件是否存在：", os.path.exists(face_path))

# 加载分类器
face_cas = cv.CascadeClassifier(face_path)
eyes_cas = cv.CascadeClassifier(eye_path)

# 检查加载状态
if face_cas.empty():
    raise RuntimeError("人脸分类器加载失败")

if eyes_cas.empty():
    raise RuntimeError("眼睛分类器加载失败")

print("人脸分类器加载成功！")
print("眼睛分类器加载成功！")
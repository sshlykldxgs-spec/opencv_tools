import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

IMG_PATH = '/Users/pengxuezhi/Desktop/huiliyi.jpg'


def detect_people(img, max_side=800, scale=1.05, min_score=0.5, nms_thresh=0.4):
    """HOG + SVM 行人检测。返回 [(x, y, w, h, 置信度), ...]（原图坐标）。没有行人返回空列表。
    相比原写法：
      1 先把图缩到长边 <= max_side：HOG 的耗时和像素数成正比，手机拍的大图能慢几十倍
      2 同时拿到置信度，丢掉低于 min_score 的（原写法忽略了置信度，什么都往外报）
      3 非极大值抑制：同一个人被多个大小不同的窗口重复框住时只留一个（原写法会出现框套框）
    注意：OpenCV 自带的这个检测器是用"站立的、全身的行人"训练的，别的东西（动漫人物、半身像、坐着的人）本来就不适用。"""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=scale)
    if len(rects) == 0:
        return []
    boxes = [[int(x), int(y), int(bw), int(bh)] for x, y, bw, bh in rects]
    scores = [float(v) for v in np.asarray(weights).ravel()]
    keep = cv.dnn.NMSBoxes(boxes, scores, min_score, nms_thresh)
    out = []
    for i in np.array(keep).flatten():
        x, y, bw, bh = boxes[i]
        out.append((int(x / s), int(y / s), int(bw / s), int(bh / s), scores[i]))
    return out


if __name__ == '__main__':
    img = cv.imread(IMG_PATH)
    if img is None:
        raise SystemExit('图片读取失败，请检查路径：' + IMG_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    hog = cv.HOGDescriptor()
    # 计算 HOG 特征向量：HOG 的标准输入是 64x128 的窗口（对整张大图直接算会得到上千万维、没法用）
    feat = hog.compute(cv.resize(gray, (64, 128)))
    print('HOG 特征维度：', feat.shape)

    # OpenCV 自带行人检测器版本（拓展）
    people = detect_people(img)
    print('检测到行人：', len(people))
    show = img.copy()
    for x, y, w, h, score in people:
        cv.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(show, f'{score:.1f}', (x, max(15, y - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(show[:, :, ::-1])
    plt.title(f'HOG 行人检测（{len(people)} 人）')
    plt.axis('off')
    plt.show()

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/Users/pengxuezhi/Desktop/huiliyi.jpg')
template = cv.imread('/Users/pengxuezhi/Desktop/moble.jpg')

if img is None or template is None:
    print("图片读取失败")
    exit()

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
tpl_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)


# ============ 方案一:SIFT 特征匹配 + RANSAC 单应性(主算法) ============
def feature_match(img_gray, tpl_gray, min_matches=10):
    # 对比度增强,让特征点更稳定
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1, g2 = clahe.apply(img_gray), clahe.apply(tpl_gray)

    sift = cv.SIFT_create(nfeatures=0, contrastThreshold=0.03)
    kp_t, des_t = sift.detectAndCompute(g2, None)
    kp_i, des_i = sift.detectAndCompute(g1, None)
    if des_t is None or des_i is None:
        return None, 0

    # FLANN + Lowe ratio test 过滤误匹配
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    knn = flann.knnMatch(des_t, des_i, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.7 * n.distance]
    print("通过比率测试的匹配点数:", len(good))
    if len(good) < min_matches:
        return None, len(good)

    src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_i[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC 求单应矩阵,剔除外点
    H, mask = cv.findHomography(src, dst, cv.USAC_MAGSAC, 5.0)
    if H is None:
        return None, len(good)
    inliers = int(mask.sum())
    print("RANSAC 内点数:", inliers)
    if inliers < min_matches:
        return None, inliers

    h, w = tpl_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    box = cv.perspectiveTransform(corners, H)   # 四边形,支持旋转/透视
    return box, inliers


# ============ 方案二:多尺度模板匹配(兜底) ============
def multiscale_match(img_gray, tpl_gray, scales=np.linspace(0.3, 2.0, 40)):
    best = (-1, None, None)  # (score, loc, (w,h))
    for s in scales:
        t = cv.resize(tpl_gray, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
        th, tw = t.shape
        if th > img_gray.shape[0] or tw > img_gray.shape[1] or th < 16 or tw < 16:
            continue
        res = cv.matchTemplate(img_gray, t, cv.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, loc, (tw, th))
    return best


# ================= 执行 =================
img_show = img.copy()
box, score = feature_match(img_gray, tpl_gray)

if box is not None:
    print("特征匹配成功,内点数:", score)
    cv.polylines(img_show, [np.int32(box)], True, (0, 0, 255), 3, cv.LINE_AA)
    title = f'SIFT+RANSAC (inliers={score})'
else:
    print("特征匹配失败,回退到多尺度模板匹配")
    mx, loc, (tw, th) = multiscale_match(img_gray, tpl_gray)
    print("多尺度最高分:", mx)
    cv.rectangle(img_show, loc, (loc[0] + tw, loc[1] + th), (0, 0, 255), 3)
    title = f'Multi-scale TM (score={mx:.3f})'

plt.imshow(img_show[:, :, ::-1])
plt.title(title), plt.xticks([]), plt.yticks([])
plt.show()

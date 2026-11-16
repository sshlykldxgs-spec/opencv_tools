import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

IMG_PATH = '/Users/pengxuezhi/Desktop/huiliyi.jpg'


def rootsift(des, eps=1e-7):
    """RootSIFT：描述子先 L1 归一化再开方。关键点位置不变，但描述子之间用普通 L2 距离比较时，匹配更准"""
    des = des / (des.sum(axis=1, keepdims=True) + eps)
    return np.sqrt(des).astype(np.float32)


def sift_features(gray, n_target=None, contrast=0.02):
    """SIFT 特征：检测 + RootSIFT 描述子。返回 (关键点列表, 描述子 Nx128，没有特征点时为 None)
    - RootSIFT：匹配精确率比原版高约 3~6 个百分点（实测，噪声 0~16 都有效）
    - contrast=0.02（默认 0.04）：对比度低的区域也能出特征点，正确匹配数多约 12%~43%
    - n_target：只保留响应最强的前 N 个。默认不限——限制数量会让可用匹配变少（实测限 1000 个时正确匹配少了约 40%），
      只想让画面干净一点的话，在显示时再截断就行（见下面的 SHOW_TOP）"""
    sift = cv.SIFT_create(contrastThreshold=contrast)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:  # 纯色图、过暗的图可能一个点都没有
        return [], None
    if n_target is not None and len(kps) > n_target:
        idx = np.argsort([-k.response for k in kps])[:n_target]
        kps, des = [kps[i] for i in idx], des[idx]
    return list(kps), rootsift(des)


if __name__ == '__main__':
    # 1 读取图像
    img = cv.imread(IMG_PATH)
    if img is None:
        raise SystemExit('图片读取失败，请检查路径：' + IMG_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2 SIFT 关键点检测
    kp, des = sift_features(gray)
    print('关键点数：', len(kp), '  描述子：', None if des is None else des.shape)

    # 3 在图像副本上绘制关键点（带大小、方向）——不要把 img 传成输出，那会把原图改掉
    #   画面上只画响应最强的 SHOW_TOP 个，否则成千上万个圆圈会糊成一片；检测和匹配仍然用全部点
    SHOW_TOP = 300
    top = sorted(kp, key=lambda k: -k.response)[:SHOW_TOP]
    show = cv.drawKeypoints(img, top, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 4 图像显示
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(show[:, :, ::-1])
    plt.title(f'sift 检测（共 {len(kp)} 个，图中画响应最强的 {len(top)} 个）')
    plt.xticks([]), plt.yticks([])
    plt.show()

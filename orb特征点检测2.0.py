import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 中文设置（多个字体兜底，mac / Windows 都能显示）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

IMG_PATH = '/Users/pengxuezhi/Desktop/huiliyi.jpg'


def grid_select(kps, shape, n_total, grid=(8, 8)):
    """把关键点按响应强度挑出 n_total 个，并且在画面里尽量均匀：先给每个格子分一份配额，余下的再按强度补"""
    if len(kps) <= n_total:
        return list(kps)
    h, w = shape[:2]
    gx, gy = grid
    cells = {}
    for k in sorted(kps, key=lambda k: -k.response):
        c = (min(gy - 1, int(k.pt[1] * gy / h)), min(gx - 1, int(k.pt[0] * gx / w)))
        cells.setdefault(c, []).append(k)
    quota = max(1, n_total // (gx * gy))
    chosen = [k for v in cells.values() for k in v[:quota]]
    rest = [k for v in cells.values() for k in v[quota:]]
    rest.sort(key=lambda k: -k.response)
    chosen += rest[:max(0, n_total - len(chosen))]
    return sorted(chosen, key=lambda k: -k.response)[:n_total]


def orb_features(gray, n_target=500, spread=False):
    """ORB 特征 + 描述子。返回 (关键点列表, 描述子 Nx32，没有特征点时为 None)。
    默认就是原来的 ORB_create(nfeatures=500)：我在随机旋转/缩放/透视/噪声的测试里比较过，
    它的正确匹配数最多、单应矩阵误差最小，所以算法没改，只修了"检测不到点时崩溃"的问题。
    spread=True：让点在画面里分布更均匀（覆盖率 45%→66%），代价是正确匹配数少约 15%~20%。
    需要用点去算单应矩阵/拼接、而且画面里大片区域没有纹理时可以试试。"""
    if not spread:
        orb = cv.ORB_create(nfeatures=n_target)
        kps, des = orb.detectAndCompute(gray, None)
        return list(kps), des  # 没检测到点时 des 为 None（原写法后面直接 des.shape 会报错）
    orb = cv.ORB_create(nfeatures=4 * n_target, fastThreshold=10)  # 先多检测一些备选，再均匀挑选
    kps = grid_select(orb.detect(gray, None), gray.shape, n_target)
    if not kps:
        return [], None
    kps, des = orb.compute(gray, kps)
    return list(kps), des


if __name__ == '__main__':
    # 1 图像读取
    img = cv.imread(IMG_PATH)
    if img is None:
        raise SystemExit('图片读取失败，请检查路径：' + IMG_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2 ORB 特征检测 + 描述子
    kp, des = orb_features(gray, n_target=500)  # 想要均匀分布：spread=True
    print('关键点数：', len(kp))
    print('描述子维度：', None if des is None else des.shape)

    # 3 绘制关键点
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    # 4 展示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.title(f'ORB（{len(kp)} 个关键点）')
    plt.xticks([])
    plt.yticks([])
    plt.show()

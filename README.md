# opencv_tools

OpenCV（Python）学习与实战脚本集合：从图像基础操作、滤波与边缘检测，到特征点检测、模版匹配、人脸检测、视频追踪，再到 YOLO 训练数据准备工具链。

脚本按**学习主题**分目录、目录名带序号；同一脚本的不同版本放在同一目录内并相邻排列，方便按顺序从基础刷到实战。

## 最新：灾情监测数据预处理 v3.1

[`00_灾情监测数据预处理/`](00_灾情监测数据预处理/) 是本仓库目前最完整的一个模块，面向目标检测的数据集制作全流程（素材整理 → 标注 → 数据集生成），带异常处理、原子写入和 25 项回归测试：

```bash
cd 00_灾情监测数据预处理
python 标注前-素材整理.py --in 原始素材 --out pre_out --mode scene
# 导入 CVAT 等工具标注
python 标注后-数据集生成.py --img pre_out/images --ann 标注 --format yolo \
       --classes classes.txt --manifest pre_out/group_manifest.csv --check-only
```

详见该目录下的 `使用说明.txt`。

## 目录结构

| 目录 | 内容 | 脚本数 |
| --- | --- | --- |
| `00_灾情监测数据预处理/` | v3.1 检测数据集制作管线：素材整理 → 标注 → 数据集生成 + 回归测试 | 9 |
| `01_基础操作/` | 图像 IO、像素读写、绘图、算术运算与混合、颜色空间转换、掩膜 | 9 |
| `02_色彩与直方图/` | 直方图统计、直方图均衡化、CLAHE 自适应均衡化 | 3 |
| `03_滤波与形态学/` | 中值/均值滤波、腐蚀膨胀、开闭运算、顶帽黑帽 | 9 |
| `04_几何变换与金字塔/` | 旋转/平移/缩放、仿射变换、透视变换、图像金字塔 | 3 |
| `05_边缘检测/` | Sobel、Scharr、Laplacian、Canny | 8 |
| `06_特征检测/` | Harris、Shi-Tomasi、SIFT、ORB、FAST、HOG、LBP 纹理 | 14 |
| `07_模版匹配/` | 多尺度模版匹配 + SIFT 特征预筛、单应性校正 | 2 |
| `08_霍夫检测/` | 霍夫直线、霍夫圆 | 4 |
| `09_人脸检测/` | Haar 级联人脸/眼睛检测 + `haarcascades/` 分类器 | 3 |
| `10_视频处理/` | 视频读写与保存、Meanshift/CamShift 目标追踪 | 12 |
| `11_YOLO数据准备/` | 抽帧去重、标注转换与体检、带框增强、数据集划分、letterbox、特征图合成 | 7 |

共 83 个脚本。

## 版本约定

| 命名 | 含义 |
| --- | --- |
| `xxx.py` | 基础版，最短的 API 演示 |
| `xxx2.0.py` | 增强版：中文显示兜底、BGR/RGB 显示修正、读取校验、参数化与结果标注 |
| `模版匹配3.0.py` | 进阶版：多尺度 + SIFT 特征配准 |

同名脚本的基础版与 2.0 版在目录内相邻排列，建议先看基础版再看 2.0 版。

## 目录明细

### 00_灾情监测数据预处理（v3.1）

健壮性 + 异常处理版：三个场景脚本共用同一套核心，只有 `CFG` 默认值不同。

- `标注前-素材整理.py` — 原始图片 / 视频 → 干净统一的图片集；产出 `images/`、`group_manifest.csv`（供标注后脚本 `--manifest` 使用）、`quality.csv`（每张图去留与原因）。参数：`--in --out --mode interval|scene --every --batch-size --max-side --workers --overwrite`
- `标注后-数据集生成.py` — 标注结果 → 训练数据集（`--format yolo | yolo_seg | coco | voc | mask | instmap`，检测框与实例分割都支持）；输出多个训练变体（`--variants base,tiles,aug_geo`）+ `data.yaml` + 报告；`--check-only` 先体检，异常写入 `xxx_check/issues.csv`
- `灾情监测-数据预处理3.0.py` — 通用灾情场景
- `水道漂浮物检测-数据预处理3.0.py` — 水道漂浮物检测场景
- `病虫害检测-数据预处理3.0.py` — 病虫害检测场景
- `使用说明.txt` — 完整命令行用法与流程
- `tests/` — 25 项回归测试，本目录运行 `python -m pytest tests -q`（约 1 分钟）

关键约定：退出码 `0` 成功 / `2` 预期内错误（配置、输入输出，屏幕给原因和建议）/ `130` 被中断 / `1` 未预料错误（写 `<out>_crash.log`）；结果先写 `<out>.partial`，成功后才替换 `<out>`，失败或中断不会破坏已有结果。

### 01_基础操作

- `图像的IO操作.py` — imread / imshow / imwrite 三件套，生成 `messigray.png`
- `修改像素.py` — 按坐标读写像素、通道切片
- `绘制图形.py` — 直线、矩形、圆、文字
- `算数操作.py` — 图像加减乘除
- `算数操作2.0.py` — 两张图尺寸对齐后再相加 / 加权
- `图像的混合(加 减 权重).py` — addWeighted 加权混合
- `图像的混合(加 减 权重)2.0.py` — 读取校验 + 尺寸对齐
- `掩膜.py` — 位运算与 ROI 掩膜提取
- `格式转换(hsv bgr grey).py` — cvtColor 颜色空间互转

### 02_色彩与直方图

- `直方图.py` — calcHist 统计并绘制三通道直方图
- `直方图均衡化（提高对比度）.py` — equalizeHist 灰度均衡化
- `自适应均衡化.py` — createCLAHE 局部自适应均衡化

### 03_滤波与形态学

- `中值滤波.py` — medianBlur 去椒盐噪声
- `均值滤波（椒盐）.py` — blur 对椒盐噪声的效果
- `均值滤波（高斯）.py` — blur 对高斯噪声的效果
- `侵蚀与膨胀.py` — erode / dilate 基础用法
- `侵蚀与膨胀2.0.py` — 修正 BGR/RGB 显示、补齐中文标题
- `开闭运算.py` — morphologyEx 开运算去噪、闭运算补洞
- `开闭运算2.0.py` — 核放大到 10×10，开 / 闭效果对比更明显
- `黑帽礼帽.py` — 顶帽突出亮细节、黑帽突出暗细节
- `黑帽礼帽2.0.py` — 改为灰度图计算，并注明彩色逐通道计算会失真

### 04_几何变换与金字塔

- `几何变换(旋转 平移 缩放 仿射 透射).py` — resize、warpAffine、warpPerspective
- `几何变换(旋转 平移 缩放 仿射 透射)2.0.py` — getRotationMatrix2D 旋转 + 透视变换并列演示
- `图像金字塔.py` — pyrDown / pyrUp 上下采样

### 05_边缘检测

- `边缘检测sobel（高速）.py` — Sobel 一阶梯度
- `边缘检测sobel（高速）2.0.py` — x / y 方向梯度 + magnitude 合成幅值，三联图
- `边缘检测scharr（sobel升级版）.py` — Scharr 小核高精度
- `边缘检测scharr（sobel升级版）2.0.py` — 先高斯平滑再求 Scharr，抑制噪点
- `边缘检测拉普拉斯（高精）.py` — Laplacian 二阶梯度
- `边缘检测拉普拉斯（高精）2.0.py` — 先高斯平滑再求二阶导，避免噪声被放大
- `边缘检测canny（最优）.py` — Canny 双阈值与滞后连接
- `边缘检测canny（最优）2.0.py` — Otsu 自动定阈值 + L2gradient，与固定阈值结果对比

### 06_特征检测

- `图像特征—角点- Harris.py` — cornerHarris 角点响应与阈值
- `图像特征—角点- Harris2.0.py` — 按噪声自适应高斯去噪 + 非极大值抑制 + cornerSubPix 亚像素精化
- `图像特征—角点- Tomas.py` — Shi-Tomasi goodFeaturesToTrack
- `图像特征—角点- Tomas2.0.py` — 自适应去噪 + cornerSubPix 亚像素精化
- `sift检测关键点.py` — SIFT 关键点与描述子
- `sift检测关键点2.0.py` — contrastThreshold 可调，只绘制响应最强的 TOP-N 个点
- `orb特征点检测.py` — ORB 快速特征
- `orb特征点检测2.0.py` — ORB + 网格配额选点，关键点分布更均匀
- `fast角点检测.py` — FAST 角点
- `fast角点检测2.0.py` — FAST + 网格配额选点，避免点全挤在纹理密集区
- `hog-物形检测.py` — HOGDescriptor 默认行人检测器
- `hog-物形检测2.0.py` — detectMultiScale 多尺度检测并绘制检测框
- `lbp-纹路-人脸检测.py` — 均匀 LBP 纹理图与直方图（半径 1、8 采样点）
- `lbp-纹路-人脸检测2.0.py` — 手写抗噪版 uniform LBP（与 skimage 定义一致）+ 直方图对比

> `grid_select`（按响应强度 + 网格配额选点）目前只用在 FAST 与 ORB 的 2.0 版。

### 07_模版匹配

- `模版匹配2.0.py` — SIFT 特征预筛 + 多尺度 TM_CCOEFF_NORMED 匹配
- `模版匹配3.0.py` — 多尺度匹配 + NCC 阈值判定 + SIFT/RANSAC 单应性校正

### 08_霍夫检测

- `霍夫检测 线.py` — HoughLinesP 线段检测
- `霍夫检测 线2.0.py` — Canny 预处理后做概率霍夫直线检测
- `霍夫检测 圆.py` — HoughCircles 圆检测
- `霍夫检测 圆2.0.py` — 先高斯模糊降噪，并用 minRadius / maxRadius 约束半径

### 09_人脸检测

- `人脸检测- hear特征.py` — Haar 级联人脸 / 眼睛检测
- `人脸检测- hear特征2.0.py` — 正脸 / 侧脸 / 眼睛 / 微笑多分类器叠加，缺失分类器自动跳过
- `ces.py` — 直接用 OpenCV 自带分类器目录的入门示例
- `haarcascades/` — 14 个官方 Haar 分类器（脚本优先读同级目录，缺失时回退 `cv2.data.haarcascades`）

### 10_视频处理

- `视频读写.py` — VideoCapture 读取、逐帧播放与属性查看
- `视频读写2.0.py` — 打印帧率等属性 + waitKey 退出控制
- `视频保存.py` — VideoWriter 编码器与帧率设置
- `视频保存2.0.py` — isOpened 校验写入器可用性 + release 收尾
- `视频追踪-第一帧对象-meanshift.py` — 首帧框选 ROI + Meanshift
- `视频追踪-第一帧对象-meanshift2.0.py` — 增加 calcHist / calcBackProject 反投影可视化
- `视频追踪-手动选取对象-ms.py` — 手动画框后 Meanshift 跟踪
- `视频追踪-手动选取对象-ms2.0.py` — 改用 selectROI 框选 + 反投影
- `视频追踪-camshift.py` — CamShift 自适应窗口
- `视频追踪-camshift2.0.py` — boxPoints 旋转框绘制 + 中心轨迹
- `视频追踪-camshift-手动画框.py` — 拖拽画框 + CamShift
- `视频追踪-camshift-手动画框2.0.py` — 支持 ESC 退出，绘制逻辑整理

### 11_YOLO数据准备

- `视频抽帧-清晰度去重.py` — Laplacian 清晰度筛帧 + dhash 差值哈希去重
- `标注格式转换.py` — VOC/COCO 与 YOLO txt 互转，坐标归一化
- `标注检查与可视化.py` — 越界等异常框体检并画框核对
- `数据增强-带标注框.py` — 翻转 / 缩放 / HSV 色彩抖动，标签随图同步变换
- `数据集划分.py` — 随机划分 train/val/test，生成 `data.yaml`
- `letterbox缩放.py` — 等比缩放 + padding 对齐网络输入，并记录还原参数
- `特征图合成-多通道输入.py` — Scharr / CLAHE / 梯度幅值等特征图拼成多通道输入

## 环境

```bash
pip install opencv-python numpy matplotlib
```

Python 3.8+，建议在虚拟环境中运行。

## 运行提示

1. **图片路径**：多数脚本把测试图路径写成了本机绝对路径（如 `/Users/pengxuezhi/Desktop/huiliyi.jpg`），clone 后请改成自己的图片路径，或把图片放到脚本同目录再改成相对文件名。
2. **分类器**：`09_人脸检测/` 下的脚本会自动在 `haarcascades/` 和 OpenCV 安装目录里找 XML，无需额外配置。
3. **摄像头**：`10_视频处理/` 的追踪脚本需要摄像头权限，macOS 首次运行请在「系统设置 → 隐私与安全性 → 摄像头」中授权终端/PyCharm。
4. **运行方式**：PyCharm 里直接右键 Run，或 `python "01_基础操作/图像的IO操作.py"`。

## 备注

- 仓库由个人学习过程整理而来，脚本之间互相独立，不构成包，无需 `__init__.py`。
- `.venv/`、`__pycache__/`、`.idea/workspace.xml` 已在 `.gitignore` 中排除。

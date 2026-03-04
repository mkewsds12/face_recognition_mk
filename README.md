# 人脸识别可视化 / Face Recognition Visualizer

一个基于 **dlib** 和 **PyQt5** 的人脸检测、关键点定位与特征提取可视化工具，支持图片和视频输入。

A face detection, landmark localization, and feature extraction visualization tool built with **dlib** and **PyQt5**, supporting both image and video input.

---

## 功能特性 / Features

| 功能 | Feature |
|------|---------|
| HOG 人脸检测（快速） | HOG Face Detection (fast) |
| CNN-mmod 人脸检测（更准确） | CNN-mmod Face Detection (more accurate) |
| 5 点 / 68 点人脸关键点定位 | 5-point / 68-point Facial Landmark Detection |
| 128 维人脸识别特征提取 | 128-D Face Recognition Feature Extraction |
| 图片 / 视频实时检测 | Image / Real-time Video Detection |
| 结果叠加层一键切换 | One-click Overlay Toggle |
| 滚轮缩放图像 | Scroll-wheel Zoom |
| 带框 / 原图保存 | Save Result / Original Image |
| 多线程检测，不阻塞 UI | Multi-threaded Detection, Non-blocking UI |

---

## 目录结构 / Project Structure

```
人脸识别项目/
├── 人脸识别可视化.py        # 主程序 / Main program
├── pt/                      # 模型文件目录 / Model files directory
│   ├── mmod_human_face_detector.dat              # CNN 检测模型
│   ├── shape_predictor_5_face_landmarks.dat      # 5 点关键点模型
│   ├── shape_predictor_68_face_landmarks.dat     # 68 点关键点模型
│   └── dlib_face_recognition_resnet_model_v1.dat # 人脸识别模型
└── image/                   # 示例图片目录 / Sample images directory
```

---

## 环境依赖 / Requirements

| 包 / Package | 版本建议 / Recommended Version |
|---|---|
| Python | ≥ 3.8 |
| dlib | ≥ 19.24 |
| opencv-python | ≥ 4.5 |
| numpy | ≥ 1.21 |
| PyQt5 | ≥ 5.15 |

使用 conda 环境（推荐）/ Using conda (recommended):

```bash
conda create -n cv python=3.9
conda activate cv
conda install dlib opencv numpy pyqt -y
```

---

## 模型文件下载 / Downloading Model Files

所有模型文件须放置在 `pt/` 目录下。  
All model files must be placed in the `pt/` directory.

| 文件名 / Filename | 说明 / Description | 官方下载 / Download |
|---|---|---|
| `mmod_human_face_detector.dat` | CNN 人脸检测器 / CNN face detector | [dlib model zoo](http://dlib.net/files/mmod_human_face_detector.dat.bz2) |
| `shape_predictor_5_face_landmarks.dat` | 5 点关键点 / 5-point landmarks | [dlib model zoo](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2) |
| `shape_predictor_68_face_landmarks.dat` | 68 点关键点 / 68-point landmarks | [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |
| `dlib_face_recognition_resnet_model_v1.dat` | 128 维特征识别 / 128-D recognition | [dlib model zoo](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2) |

> 下载后解压 `.bz2` 文件，将 `.dat` 文件移至 `pt/` 目录即可。  
> After downloading, extract the `.bz2` archive and move the `.dat` file into the `pt/` directory.

---

## 使用方法 / Usage

```bash
conda activate cv
python 人脸识别可视化.py
```

### 界面操作 / UI Guide

| 操作 / Action | 说明 / Description |
|---|---|
| **打开图片** / Open Image | 加载图片进行检测 / Load an image for detection |
| **打开视频** / Open Video | 加载视频进行实时检测 / Load a video for real-time detection |
| **停止视频** / Stop Video | 停止视频播放 / Stop video playback |
| **HOG / CNN 单选** / Detector Radio | 切换检测器 / Switch face detector |
| **关键点模型下拉** / Landmark Combo | 选择 5 点或 68 点模型 / Select 5-pt or 68-pt model |
| **启用特征提取** / Enable Recognition | 开启 128 维特征提取 / Enable 128-D feature extraction |
| **显示/隐藏结果** / Toggle Overlay | 一键隐藏或显示识别结果 / Show or hide detection results |
| **滚轮** / Mouse Wheel | 缩放图像 / Zoom image |
| **保存带框 / 保存原图** / Save | 保存检测结果或原图 / Save result or original image |

---

## 实现原理 / How It Works

```
输入帧 (Input Frame)
      │
      ├─ HOG 检测器 ──→ 人脸矩形列表
      └─ CNN 检测器 ──→ 人脸矩形列表
                │
                ▼
      shape_predictor → 关键点坐标 (5 or 68 points)
                │
                ▼
      face_recognition_model → 128 维特征向量
                │
                ▼
      绘制结果叠加层 → 显示到 QGraphicsView
```

- 检测在独立 `QThread` 中执行，保证 UI 主线程流畅。  
  Detection runs in a dedicated `QThread` to keep the UI responsive.
- 视频模式下自动丢弃积压帧，保证实时性。  
  In video mode, backlogged frames are dropped automatically to maintain real-time performance.
- 使用 `numpy.fromfile` / `cv2.imdecode` 兼容中文路径。  
  Chinese file paths are handled via `numpy.fromfile` / `cv2.imdecode`.

---

## 许可证 / License

本项目仅供学习和研究使用，不得用于任何商业用途。  
This project is for educational and research purposes only and may not be used for any commercial purposes.

# 🪖 YOLOv11 安全帽佩戴检测系统

基于 **YOLOv11s** 的施工现场安全帽佩戴检测系统，支持实时摄像头检测、图片检测、视频文件检测，并提供基于 PyQt5 的可视化图形界面。

---

## ✨ 功能特性

- 🎯 **模型训练** — 基于 YOLOv11s 微调训练，支持自定义数据集
- 📷 **实时检测** — 摄像头实时检测、图片检测、视频逐帧检测
- 🖥️ **图形界面** — PyQt5 可视化界面，支持中文标签显示
- 🔵🔴 **颜色区分** — 蓝色框 = 佩戴安全帽，红色框 = 未戴安全帽
- 📊 **实时统计** — FPS、检测数量、分类统计实时更新
- 💾 **结果保存** — 截图保存、视频录制、历史记录管理

---

## 📦 数据集

> ⚠️ **重要说明**：本仓库中 **不包含完整数据集**，仅提供了部分示例数据用于验证流程。完整数据集请从以下地址获取：

🔗 **完整数据集**：[Kaggle - Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)

### 数据集格式

本项目采用 YOLO 格式标注，目录结构如下：

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 标注格式

每行一个目标框，坐标为归一化值（0~1）：

```
class_id center_x center_y width height
```

- `0` — helmet（佩戴安全帽）
- `1` — no_helmet（未戴安全帽）

### 配置文件 `dataset/data.yaml`

```yaml
path: ./dataset
train: train/images
val: val/images
test: test/images
nc: 2
names: ['helmet', 'no_helmet']
```

### 使用完整数据集

1. 从 Kaggle 下载完整数据集
2. 将原始标注转换为 YOLO 格式（Pascal VOC → YOLO）
3. 按 train/val/test ≈ 70%/15%/15% 划分
4. 放入 `dataset/` 目录下对应文件夹

---

## 💻 环境要求

- **操作系统**：Windows 10/11 (64位)
- **Python**：3.8+
- **显卡**：NVIDIA GPU（推荐 4GB+ 显存）

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install ultralytics>=8.3.0 opencv-python>=4.5.0 PyQt5>=5.15.0 numpy>=1.20.0 pandas>=1.3.0 Pillow>=9.0.0 torch>=2.0.0
```

---

## 🚀 快速开始

### 方式一：直接使用 GUI（推荐）

前提：确保已训练好模型，权重文件位于 `runs/detect/helmet_final_phase/weights/best.pt`。

```bash
python qt界面.py
```

### 方式二：完整流程

1. 准备数据集 → 2. 训练模型 → 3. 评估模型 → 4. 启动 GUI

---

## 🎓 模型训练

```bash
python 训练.py
```

### 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| 模型 | yolo11s.pt | YOLOv11s 预训练权重 |
| epochs | 50 | 训练轮数 |
| imgsz | 640 | 输入图像尺寸 |
| batch | 4 | 批次大小（根据显存调整） |
| device | 0 | GPU 设备 |
| lr0 | 0.001 | 初始学习率 |
| optimizer | AdamW | 优化器 |
| patience | 15 | 早停耐心值 |
| amp | True | 混合精度训练 |

### 训练输出

训练完成后，结果保存在 `runs/detect/helmet_final_phase/`：

- `weights/best.pt` — **最佳模型权重**（验证集 mAP50 最高时保存）
- `weights/last.pt` — 最后一个 epoch 的权重
- `results.csv` — 每轮训练指标记录
- `results.png` — 训练曲线可视化
- `confusion_matrix.png` — 混淆矩阵
- `BoxPR_curve.png` — PR 曲线

---

## 📊 模型评估

```bash
python 测试集上进行正式评估.py
```

### 评估结果

| 指标 | 数值 | 说明 |
|------|------|------|
| **mAP50** | **95.25%** | IoU=0.5 时的平均精度 |
| **mAP50-95** | 64.58% | IoU=0.5:0.95 的平均精度 |
| **Precision** | 91.40% | 精确度 |
| **Recall** | 91.05% | 召回率 |

### 各类别性能

| 类别 | mAP50 |
|------|-------|
| helmet（安全帽） | 96.12% |
| no_helmet（未戴） | 94.34% |

### 推理速度参考

| 设备 | FPS |
|------|-----|
| NVIDIA RTX 3060 | ~30-45 |
| NVIDIA GTX 1660 | ~20-30 |
| CPU (i7-10700) | ~5-10 |

---

## 🖥️ GUI 界面

```bash
python qt界面.py
```

### 界面功能

| 功能 | 操作 | 说明 |
|------|------|------|
| 📷 开启摄像头 | 点击按钮 | 实时检测，支持截图和录制 |
| 🖼️ 选择图片 | 点击按钮 | 支持 jpg/png/bmp |
| 🎬 选择视频 | 点击按钮 | 支持 mp4/avi/mov/mkv/wmv，可拖动进度条 |
| 💾 保存当前 | 点击按钮 | 保存当前帧截图 |
| ⏺️ 开始录制 | 点击按钮 | 录制带标注框的视频 |
| 置信度滑块 | 拖动调节 | 范围 0.10~0.90，默认 0.25 |

### 检测颜色规范

- 🔵 **蓝色框** — 佩戴安全帽（安全状态）
- 🔴 **红色框** — 未戴安全帽（危险状态）

### 保存目录

| 模式 | 截图保存路径 | 录制保存路径 |
|------|-------------|-------------|
| 摄像头 | `saved_results/camera/` | `saved_results/camera_recordings/` |
| 图片 | `saved_results/images/` | — |
| 视频 | `saved_results/video_snapshots/` | `saved_results/video_recordings/` |

---

## 📁 项目结构

```
yolo11 安全帽/
├── 训练.py                          # 模型训练脚本
├── qt界面.py                        # PyQt5 图形界面程序
├── 测试集上进行正式评估.py             # 测试集评估脚本
├── requirements.txt                 # 依赖列表
├── yolo11s.pt                       # YOLOv11s 预训练权重
├── yolo11n.pt                       # YOLOv11n 预训练权重
├── dataset/
│   ├── data.yaml                    # 数据集配置
│   ├── filename_mapping.csv         # 文件名映射
│   ├── train/images/ & labels/      # 训练集
│   ├── val/images/ & labels/        # 验证集
│   └── test/images/ & labels/       # 测试集
├── runs/detect/
│   ├── helmet_final_phase/weights/  # 训练模型权重
│   └── final_test_evaluation/       # 测试评估结果
└── saved_results/                   # 检测结果保存（运行后生成）
```

---

## ❓ 常见问题

<details>
<summary>启动 GUI 提示找不到模型文件？</summary>

确保 `runs/detect/helmet_final_phase/weights/best.pt` 存在。先运行 `python 训练.py` 完成训练。
</details>

<details>
<summary>训练时显存不足（CUDA OOM）？</summary>

- 减小 batch size：`batch=2` 或 `batch=1`
- 减小图像尺寸：`imgsz=480`
- 关闭混合精度：`amp=False`
</details>

<details>
<summary>中文标签显示为方框？</summary>

确保系统存在中文字体：`C:/Windows/Fonts/simhei.ttf` 或 `C:/Windows/Fonts/msyh.ttc`。
</details>

<details>
<summary>摄像头无法打开？</summary>

- 检查摄像头权限
- 关闭其他占用摄像头的程序
- 尝试修改代码中的摄像头索引号
</details>

---

## 📄 License

本项目仅供学习研究使用。数据集遵循 [Kaggle Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) 的原始许可协议。

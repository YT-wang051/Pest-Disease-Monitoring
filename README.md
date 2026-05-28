# 🐛 病虫害监测项目 

## 📌 项目简介
本项目基于 Ultralytics YOLOv11 开发，用于实现农业场景下的**病虫害目标检测**，支持实时推理、视频流检测与批量图像分析，可部署在本地、边缘设备或服务器环境中。

## 📁 项目结构
ultralytics-8.3.39/
├── docs/ # 项目文档与说明
├── examples/ # 功能示例代码（含 Jupyter Notebook 教程）
├── tests/ # 单元测试与功能验证脚本
├── ultralytics/ # YOLOv11 核心源码
├── data.yaml # 数据集配置文件
├── train.py # 模型训练脚本
├── detect.py # 目标检测推理脚本
├── split.py # 数据集划分工具
└── ...

## 🚀 快速开始

### 1. 环境安装
```bash
# 安装依赖
pip install -e .

###2.模型训练
python train.py --data data.yaml --epochs 100 --batch 16
###3.目标检测推理
# 单张图片检测
python detect.py --source test.jpg --weights runs/train/exp/weights/best.pt

# 视频流检测
python detect.py --source video.mp4 --weights runs/train/exp/weights/best.pt
✨ 核心功能
支持病虫害多类别目标检测
兼容图像、视频、摄像头流等多种输入源
提供训练、验证、推理全流程工具链
支持 ONNX、TensorRT 等多种部署格式

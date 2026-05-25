<div align="center">
 STDDF

**Spatio-temporal Aware Multi-scale Feature Flow**

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img width="907" height="434" alt="1" src="https://github.com/user-attachments/assets/4e153254-0417-40f3-99e4-09f2280f9826" />

</div>
##  目录

- [环境配置](#-环境配置)
- [训练及测试](#-训练及测试)
- [数据集格式](#-数据集格式)

---

##  环境配置

### Step 1: 创建环境

```bash
conda create -n STDDF python=3.8
conda activate STDDF
```

### Step 2: 安装依赖

```bash
pip install -r requirements.txt
```

---

##  训练及测试

### 训练命令

以 GZ-CD 数据集为例：

```bash
python train.py --config/gzcd.json
```

### 测试命令

```bash
python test.py --config/gzcd_test.json
```
### 测试权重及日志

```bash
通过网盘分享的文件：checkpoint.zip 链接: https://pan.baidu.com/s/1l297KTQHNqUQ7hAq0HXySA?pwd=hp2y 提取码: hp2y
```
### 参数说明

| 参数 | 描述 |
|------|------|
| `--config` | 配置文件路径 |
### 实验结果
<img width="1457" height="662" alt="7" src="https://github.com/user-attachments/assets/1fd9a085-e063-48ab-9d3e-44c6cbe0c3da" />
<img width="1459" height="664" alt="6" src="https://github.com/user-attachments/assets/db6e9282-67b6-401c-900c-79b3209687cd" />
<img width="1456" height="662" alt="5" src="https://github.com/user-attachments/assets/1b71c0ec-929e-42bf-a534-32a3d3080bfe" />



---

##  数据集格式

### 目录结构

数据集应按以下结构组织：

```
GZ-CD/
├── A/
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├── ...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├── ...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── B/
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├── ...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├── ...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── label/
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├── ...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├── ...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

<div align="center">

** Remote Sensing Change Detection**

</div>

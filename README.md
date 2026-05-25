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
![图7](https://github.com/user-attachments/assets/8b0dfbdf-18f0-451c-9bd2-9520aa1fc4d1)  
![图8](https://github.com/user-attachments/assets/ade6bf44-3709-4984-ab05-f704cafd6376)
![图9](https://github.com/user-attachments/assets/4ffac8ce-89e9-4467-9f76-101d696eb8f2)


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

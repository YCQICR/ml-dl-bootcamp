# 一个月 ML/DL 快速入门（科研导向版）

> 目标：4 周、每周 20–30 小时，达到"能独立用 PyTorch 完成 **数据 → 训练 → 评估** 全流程，能训练并调优 CNN，能读懂一篇论文的方法部分"。
> 前置：Python 基础、数学较扎实（高数/线代/概率）、科研导向、方向未定。

## 四周路线总览

| 周 | 主题 | 主要产出 | 验收标准 |
|---|---|---|---|
| 第 1 周 | 工具 + 机器学习直觉 | 环境、Numpy/Pandas/Matplotlib 练习、sklearn 小项目 | 能解释过拟合与偏差-方差 |
| 第 2 周 | PyTorch 与深度学习核心 | 手写反向传播、MLP 训练 Fashion-MNIST > 85% | 能手推 2 层网络反向传播 |
| 第 3 周 | CNN 实战 | CIFAR-10 训练 + 至少 2 组对比实验 | 从零训练 CNN 测试精度 > 70% |
| 第 4 周 | 论文阅读与收尾 | ResNet 论文卡片、GitHub 仓库、一页实验报告 | 15 分钟讲清 ResNet 的方法 |

## 目录结构

```text
ml-dl-bootcamp/
├── README.md                 # 本文件：总路线与使用说明
├── requirements.txt          # Python 依赖
├── data/                     # 数据目录（Titanic CSV、torchvision 数据集）
├── week1/                    # 工具 + 机器学习直觉
├── week2/                    # PyTorch 核心 + 手写反向传播 + MLP
├── week3/                    # CNN 实战：LeNet / ResNet + CIFAR-10
├── week4/                    # 论文阅读 + Transformer 背景 + 实验报告
├── checkpoints/              # 训练中自动保存的最佳模型（运行时生成）
└── runs/                     # TensorBoard 日志（运行时生成）
```

## 快速开始

1. 按 [week1/setup_guide.md](week1/setup_guide.md) 安装 Miniconda、PyTorch（CUDA 版）和本项目依赖。
2. 每周先读 `weekN/README.md`，再运行对应脚本；代码中有详细中文注释，先读注释再运行。
3. 每个脚本都支持 `--help` 查看参数；`--quick` / `--synthetic` 用于快速验证或没有网络/GPU 的环境。
4. 每完成一个任务，在下方进度表打勾。

```powershell
# 进入项目目录
cd D:\ml-dl-bootcamp
# 运行第 1 周练习
python week1\numpy_pandas_plot.py
python week1\titanic_ml.py
```

## 学习节奏建议

- 每周 5–6 天，每天 3–4 小时；时间分配约 40% 概念/论文、40% 代码、20% 笔记。
- 视频课（吴恩达、李沐）建议 1.25–1.5 倍速，**重点跟着做代码而不是只"看懂了"**。
- 每完成一个脚本，先运行、再改参数重跑一遍，最后把结果写进 `week4/report_template.md` 式的笔记。

## 进度跟踪

| 周 | 任务 | 完成 |
|---|---|---|
| 1 | 环境搭建完成，`python -c "import torch; print(torch.cuda.is_available())"` 输出 True | ☐ |
| 1 | 跑通 `numpy_pandas_plot.py` 并看懂每段代码 | ☐ |
| 1 | 跑通 `titanic_ml.py`，能讲出 train/val/test 为什么分开 | ☐ |
| 2 | 跑通 `pytorch_basics.py` | ☐ |
| 2 | 跑通 `backprop_from_scratch.py`，梯度校验通过 | ☐ |
| 2 | `mlp_fashion_mnist.py` 测试精度 > 85% | ☐ |
| 2 | 跑通 `optimizers_demo.py`，能解释 SGD/Momentum/Adam 区别 | ☐ |
| 3 | `lenet_cifar10.py` 训练完成并出曲线 | ☐ |
| 3 | `resnet_cifar10.py` 测试精度 > 70% | ☐ |
| 3 | 完成 `experiments.md` 里的 2 组对比实验并填表 | ☐ |
| 4 | 用三遍法读完 ResNet 论文，写好论文卡片 | ☐ |
| 4 | 读完《The Illustrated Transformer》并整理笔记 | ☐ |
| 4 | 完成一页实验报告 | ☐ |
| 4 | 代码整理进 GitHub 并写清 README | ☐ |

## 验收清单（一个月结束时的标准）

- [ ] 能手推 2 层 MLP 反向传播（`backprop_from_scratch.py` 的梯度对比通过）
- [ ] Fashion-MNIST 分类 > 85%
- [ ] CIFAR-10 CNN > 70%，且完成至少 2 组对比实验并解释结果
- [ ] 能用"问题 / 方法 / 结果"三要素讲清 ResNet 论文
- [ ] GitHub 仓库可复现：README + 训练曲线 + 最终指标

## 核心资源

- 吴恩达《机器学习》：Coursera 或 B 站中文版（第 1 周选看）
- 李沐《动手学深度学习》（d2l）：<https://zh-v2.d2l.ai/>
- 3Blue1Brown 神经网络系列：<https://www.3blue1brown.com/topics/neural-networks>
- ResNet 论文：<https://arxiv.org/abs/1512.03385>
- The Illustrated Transformer：<https://jalammar.github.io/illustrated-transformer/>

## 已实测通过（2026-08-11）

- 环境：`ml-dl`（Python 3.11），PyTorch `2.6.0+cu124`，RTX 3060 Laptop 6GB，CUDA 可用。
- 已验证脚本：第 1 周两个脚本、第 2 周全部脚本、第 3 周 LeNet 与 ResNet-18
  （`--quick --synthetic` 快速模式跑通全流程）。
- GPU 实测：ResNet-18 在合成数据上 1 个 epoch 约 3.3 秒（CPU 约 69 秒）。
- 说明：快速模式用合成数据验证流程；正式训练时脚本会自动下载真实数据集。

## 本月明确不做

- RNN/LSTM、Transformer 完整实现、大模型微调（留到入学后）
- SVM/聚类/PCA 的细节与深度推导
- 模型部署、MLOps、论文中的证明细节
- 方向选择（CV / NLP / 多模态等），一个月结束后再"抽样选定"

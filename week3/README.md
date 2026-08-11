# 第 3 周：CNN 实战（CIFAR-10）

> 预计投入：25–30 小时（d2l CNN 章节 12h + CIFAR-10 项目 15h）
> 验收标准：从零训练 CNN 在 CIFAR-10 上测试精度 > 70%；完成至少 2 组对比实验并解释结果。

## 本周任务

- [ ] 配套观看 d2l 第 7 章（卷积神经网络）和第 8 章（现代 CNN），理解卷积、池化、ResNet
- [ ] 运行 `lenet_cifar10.py --quick` 验证流程
- [ ] 正式训练 LeNet（`--epochs 20`），记录精度
- [ ] 正式训练 ResNet-18（`--epochs 30`），目标 > 70%（带增强时通常可达 85%+）
- [ ] 完成 `experiments.md` 里的 2 组对比实验并填表
- [ ] 用 TensorBoard 查看曲线：`tensorboard --logdir runs`

## 运行方式

```powershell
cd D:\ml-dl-bootcamp
conda activate ml-dl

# 快速验证（合成数据）
python week3\lenet_cifar10.py --quick --synthetic --epochs 1
python week3\resnet_cifar10.py --quick --synthetic --epochs 1

# 正式训练（第一次会下载 CIFAR-10，约 160MB）
python week3\lenet_cifar10.py --epochs 20
python week3\resnet_cifar10.py --epochs 30

# 查看训练曲线
tensorboard --logdir runs
```

## 本周代码

| 文件 | 作用 |
|---|---|
| `data_utils.py` | 加载 CIFAR-10（可选数据增强），无网络时退化为合成数据 |
| `lenet_cifar10.py` | 从零实现 LeNet-5 并训练 |
| `resnet_cifar10.py` | ResNet-18/34 + 数据增强 + 早停 + 最佳模型保存 |
| `experiments.md` | 2 组对比实验的操作说明与结果表 |

## 检查自己是否真的懂了

1. 卷积核的"局部性"和"权值共享"分别解决什么问题？
2. ResNet 的残差连接为什么能训练更深的网络？
3. 数据增强为什么能缓解过拟合？它等价于什么？
4. 对比实验里只允许改一个变量，为什么？

# CIFAR-10 对比实验

目标：通过**每次只改一个变量**，理解什么影响精度。所有实验用同一命令模板，
结果填进下表，并写一段"为什么"。

## 实验 1：数据增强（开 vs 关）

```powershell
python week3\resnet_cifar10.py --epochs 30                 # 实验 A：开增强（默认）
python week3\resnet_cifar10.py --epochs 30 --no-augment    # 实验 B：关增强
```

| 实验 | 数据增强 | 最佳验证精度 | 测试精度 | 训练曲线特征 |
|---|---|---|---|---|
| A | 开 | | | 训练 loss 下降慢，val 与 train 差距小 |
| B | 关 | | | 训练 loss 降得快，val 可能回升（过拟合） |

思考：数据增强 = 用裁剪/翻转"造出更多训练样本"，等价于正则化，抑制过拟合。

## 实验 2：学习率（0.1 / 0.05 / 0.01）

```powershell
python week3\resnet_cifar10.py --epochs 30 --lr 0.1
python week3\resnet_cifar10.py --epochs 30 --lr 0.05
python week3\resnet_cifar10.py --epochs 30 --lr 0.01
```

| 学习率 | 最佳验证精度 | 测试精度 | 观察 |
|---|---|---|---|
| 0.1 | | | 可能震荡，loss 下降不稳定 |
| 0.05 | | | 默认值，通常稳定 |
| 0.01 | | | 收敛慢，可能没跑够就早停 |

思考：学习率太小收敛慢，太大震荡。SGD 的常用学习率比 Adam 大一个量级。

## 可选实验 3：网络深度（18 vs 34）

```powershell
python week3\resnet_cifar10.py --epochs 30 --depth 34
```

| 深度 | 参数量 | 最佳验证精度 | 测试精度 |
|---|---|---|---|
| 18 | 约 11M | | |
| 34 | 约 21M | | |

思考：ResNet 出现前，网络加深会退化（训练误差反而更高）；残差连接为什么能解决？

## 实验记录原则（科研习惯）

1. 每次实验只改一个变量，其他参数保持一致。
2. 固定随机种子（脚本默认 `--seed 42`），保证可复现。
3. 记录：命令、最终指标、曲线、当时的直觉判断。
4. 把结果写进 `week4/report_template.md`，一周结束做一次总结。

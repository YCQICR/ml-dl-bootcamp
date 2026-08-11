# 第 2 周：PyTorch 与深度学习核心

> 预计投入：25–30 小时（PyTorch 基础 6h + d2l 核心章节 15h + 训练技巧 6h）
> 验收标准：不看书能手推一次 2 层网络反向传播；Fashion-MNIST 测试精度 > 85%。

## 本周任务

- [ ] 运行 `pytorch_basics.py`，看懂张量、autograd、Dataset/DataLoader
- [ ] 运行 `linear_regression_from_scratch.py`（手动梯度 vs autograd 对比）
- [ ] 运行 `softmax_regression.py`（手动 softmax + 交叉熵）
- [ ] 运行 `backprop_from_scratch.py`，梯度校验通过（这是本周最重要的一步）
- [ ] 运行 `mlp_fashion_mnist.py`，测试精度 > 85%
- [ ] 运行 `optimizers_demo.py`，能解释 SGD / Momentum / Adam 的区别
- [ ] 配套观看 d2l 教材第 3–6 章（线性回归、softmax、MLP、反向传播）

## 运行方式

```powershell
cd outputs\ml-dl-bootcamp
conda activate ml-dl

# 快速验证（小数据、少 epoch）
python week2\pytorch_basics.py
python week2\backprop_from_scratch.py
python week2\softmax_regression.py --quick --epochs 3
python week2\mlp_fashion_mnist.py --quick --epochs 5
python week2\optimizers_demo.py --quick --epochs 2

# 正式训练（下载 Fashion-MNIST，第一次需要联网）
python week2\mlp_fashion_mnist.py --epochs 10
```

## 本周代码

| 文件 | 作用 |
|---|---|
| `data_utils.py` | 加载 Fashion-MNIST；无网络时自动退化为合成数据 |
| `pytorch_basics.py` | 张量 / autograd / Dataset / 训练循环 |
| `linear_regression_from_scratch.py` | d2l 风格从零实现 + 手动梯度校验 |
| `softmax_regression.py` | 手动 softmax + 交叉熵，及 nn 简洁版 |
| `backprop_from_scratch.py` | **核心里程碑**：手写 2 层 MLP 反向传播 |
| `mlp_fashion_mnist.py` | MLP 训练 Fashion-MNIST（目标 > 85%） |
| `optimizers_demo.py` | SGD / Momentum / Adam / L2 / Dropout 对比 |

## 检查自己是否真的懂了

1. 为什么反向传播比"对每个参数求数值梯度"快？
2. `loss.backward()` 之后，`param.grad` 存的是什么？为什么更新前要 `zero_grad()`？
3. softmax + 交叉熵为什么要一起用？数值上有什么好处？
4. Adam 和 SGD 的核心区别是什么？什么时候选哪个？

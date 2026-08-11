"""
第 2 周 · 练习 1：PyTorch 基础
================================
运行: python week2/pytorch_basics.py

内容：
1. 张量：创建、形状、广播、索引、矩阵运算
2. autograd：自动求导，与手算梯度对比
3. Dataset / DataLoader：数据管道
4. 一个最小的训练循环（线性回归）
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", DEVICE)


def part1_tensors():
    print("\n" + "=" * 60)
    print("1) 张量基础")
    print("=" * 60)

    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("a:", a.shape, a.dtype)
    print("reshape(3,2):\n", a.reshape(3, 2))
    print("a + 10（广播）:\n", a + 10)
    print("a[0] 第一行:", a[0])
    print("a[:, 1:] 切片:\n", a[:, 1:])

    x = torch.randn(4, 3)
    w = torch.randn(3, 5)
    print("矩阵乘法 (4,3) @ (3,5) ->", (x @ w).shape)

    t = torch.tensor([1, 2, 3])
    print("sum:", t.sum().item(), "| mean:", t.float().mean().item())


def part2_autograd():
    print("\n" + "=" * 60)
    print("2) autograd：自动求导")
    print("=" * 60)

    # 例：y = x^2 + 3x + 1，在 x=2 处导数 = 2*2 + 3 = 7
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    print("x.grad（应为 7）:", x.grad.item())

    # 线性回归梯度：手动解析式 vs autograd
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    y = X @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.1
    w = torch.randn(3, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    y_hat = X @ w + b
    loss = ((y_hat - y) ** 2).mean()
    loss.backward()

    n = X.shape[0]
    # 标准 MSE: loss = mean((y_hat - y)^2)
    # 其导数: dL/dw = 2/n * X^T (y_hat - y)，dL/db = 2/n * sum(y_hat - y)
    residual = X @ w.detach() + b.detach() - y
    grad_w_manual = 2 * (X.T @ residual) / n
    grad_b_manual = 2 * residual.sum() / n
    print("dL/dw 手算:", grad_w_manual.flatten().tolist())
    print("dL/dw autograd:", w.grad.flatten().tolist())
    print("dL/db 手算:", grad_b_manual.item(), "| autograd:", b.grad.item())


def part3_dataset():
    print("\n" + "=" * 60)
    print("3) Dataset / DataLoader")
    print("=" * 60)

    class TinyDataset(Dataset):
        def __init__(self, n=32, d=4):
            self.X = torch.randn(n, d)
            self.y = (self.X[:, 0] > 0).long()

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    loader = DataLoader(TinyDataset(), batch_size=8, shuffle=True)
    for i, (xb, yb) in enumerate(loader):
        print(f"batch {i}: X={tuple(xb.shape)} y={tuple(yb.shape)}")
        if i == 2:
            break


def part4_training_loop():
    print("\n" + "=" * 60)
    print("4) 最小训练循环（线性回归）")
    print("=" * 60)

    torch.manual_seed(0)
    true_w = torch.tensor([[2.0], [-3.4]])
    true_b = 4.2
    X = torch.randn(1000, 2)
    y = X @ true_w + true_b + torch.randn(1000, 1) * 0.01

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    losses = []
    for epoch in range(5):
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        losses.append(epoch_loss / len(X))
        print(f"epoch {epoch + 1}: loss={losses[-1]:.5f}")

    w_hat, b_hat = model.weight.detach().flatten(), model.bias.detach().item()
    print("学到 w:", w_hat.tolist(), "（真实 [2.0, -3.4]）")
    print("学到 b:", round(b_hat, 3), "（真实 4.2）")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, 6), losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Minimal training loop")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "pytorch_basics_loss.png"
    fig.savefig(out, dpi=150)
    print("已保存:", out)


if __name__ == "__main__":
    part1_tensors()
    part2_autograd()
    part3_dataset()
    part4_training_loop()
    print("\n完成！回答 README 里的思考题，并进入 backprop_from_scratch.py。")

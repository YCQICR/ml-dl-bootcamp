"""
第 2 周 · 练习 3：softmax 回归（分类）
=======================================
运行:
    python week2/softmax_regression.py --quick --epochs 3   # 快速验证
    python week2/softmax_regression.py --epochs 5           # 正式训练

内容：
1. 手动实现 softmax + 交叉熵，autograd 负责反向传播
2. torch.nn 简洁版对照
"""
import argparse
import sys
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


class SoftmaxRegressionScratch(nn.Module):
    """手动 softmax 回归：W 和 b 是普通张量，靠 autograd 求梯度。"""

    def __init__(self, num_inputs, num_outputs, sigma=0.01, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.W = nn.Parameter(torch.normal(0, sigma, (num_inputs, num_outputs)))
        self.b = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        flat = X.reshape(-1, self.W.shape[0])
        return torch.softmax(flat @ self.W + self.b, dim=1)

    def params(self):
        return [self.W, self.b]


def cross_entropy_manual(y_hat, y):
    """手动交叉熵（d2l 风格）。加 1e-8 防止 log(0)。"""
    return -torch.log(y_hat[range(len(y_hat)), y] + 1e-8)


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def sgd_manual(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_scratch(train_loader, test_loader, epochs, lr, device):
    model = SoftmaxRegressionScratch(num_inputs=28 * 28, num_outputs=10).to(device)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(epochs):
        total_loss, total_correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            y_hat = model(Xb)
            loss = cross_entropy_manual(y_hat, yb).mean()
            loss.backward()
            sgd_manual(model.params(), lr, Xb.shape[0])
            total_loss += loss.item() * len(yb)
            total_correct += (y_hat.argmax(1) == yb).sum().item()
            total += len(yb)
        test_acc = evaluate(model, test_loader, device)
        history["loss"].append(total_loss / total)
        history["train_acc"].append(total_correct / total)
        history["test_acc"].append(test_acc)
        print(f"[scratch] epoch {epoch + 1}: loss={history['loss'][-1]:.4f} "
              f"train_acc={history['train_acc'][-1]:.4f} test_acc={test_acc:.4f}")
    return history


def train_concise(train_loader, test_loader, epochs, lr, device):
    net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(epochs):
        total_loss, total_correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            trainer.zero_grad()
            logits = net(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            trainer.step()
            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        test_acc = evaluate(net, test_loader, device)
        history["loss"].append(total_loss / total)
        history["train_acc"].append(total_correct / total)
        history["test_acc"].append(test_acc)
        print(f"[torch.nn] epoch {epoch + 1}: loss={history['loss'][-1]:.4f} "
              f"train_acc={history['train_acc'][-1]:.4f} test_acc={test_acc:.4f}")
    return history


def evaluate(model, loader, device):
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            correct += (model(Xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


def plot(history, title, path):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, history["loss"], marker="o")
    axes[0].set_title(f"{title}: loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["test_acc"], marker="s", label="test")
    axes[1].set_title(f"{title}: accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("已保存:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1,
                        help="手写 SGD 版的学习率（torch.nn 版用 Adam，固定 1e-3）")
    parser.add_argument("--quick", action="store_true", help="只用 5000 个样本")
    parser.add_argument("--synthetic", action="store_true", help="用随机数据")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quick = 5000 if args.quick else None
    train_loader, test_loader, *_ = data_utils.load_fashion_mnist(
        batch_size=args.batch_size, quick=quick, synthetic=args.synthetic)

    h1 = train_scratch(train_loader, test_loader, args.epochs, args.lr, device)
    plot(h1, "scratch softmax", FIG_DIR / "softmax_scratch.png")
    # Adam 常用学习率是 1e-3（比 SGD 小一个量级）
    h2 = train_concise(train_loader, test_loader, args.epochs, 1e-3, device)
    plot(h2, "torch.nn softmax", FIG_DIR / "softmax_concise.png")


if __name__ == "__main__":
    main()

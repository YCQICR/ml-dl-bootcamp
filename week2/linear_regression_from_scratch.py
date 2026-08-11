"""
第 2 周 · 练习 2：线性回归从零实现（d2l 风格）
=================================================
运行: python week2/linear_regression_from_scratch.py

内容：
1. 手写数据迭代器、模型、损失、SGD 更新
2. 解析梯度 vs autograd 对比（理解 backward 在算什么）
3. torch.nn 简洁版对照
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRUE_W = torch.tensor([2.0, -3.4])
TRUE_B = 4.2


def synthetic_data(w, b, num_examples=1000):
    """生成 y = Xw + b + 噪声 的合成数据。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b + torch.normal(0, 0.01, (num_examples,))
    return X, y.reshape(-1, 1)


def data_iter(batch_size, features, labels):
    n = len(features)
    indices = torch.randperm(n)
    for i in range(0, n, batch_size):
        batch_indices = indices[i: i + batch_size]
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    return X @ w + b


def squared_loss(y_hat, y):
    return ((y_hat - y) ** 2) / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def gradient_check(features, labels, batch_size=32):
    """解析梯度 vs autograd 对比。"""
    print("\n[梯度校验] 解析式 dL/dw = X^T*(Xw+b-y)（损失为 sum(0.5*(y_hat-y)^2)，SGD 里再除以 batch_size）")
    Xb, yb = next(iter(data_iter(batch_size, features, labels)))
    w = torch.normal(0, 0.01, (features.shape[1], 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    y_hat = Xb @ w + b
    grad_w_analytic = Xb.T @ (y_hat - yb)
    grad_b_analytic = (y_hat - yb).sum()

    loss = squared_loss(y_hat, yb).sum()
    loss.backward()

    print("dL/dw 解析:", grad_w_analytic.flatten().tolist())
    print("dL/dw autograd:", w.grad.flatten().tolist())
    print("dL/db 解析:", grad_b_analytic.item(), "| autograd:", b.grad.item())
    ok = torch.allclose(grad_w_analytic, w.grad, atol=1e-6)
    print("梯度一致:", ok)


def train_scratch(features, labels, batch_size=32, lr=0.03, epochs=5):
    """从零实现的训练循环。"""
    w = torch.normal(0, 0.01, (features.shape[1], 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for Xb, yb in data_iter(batch_size, features, labels):
            # d2l 约定：损失是 Σ 0.5(ŷ-y)²，sgd() 内部再除以 batch_size
            loss = squared_loss(linreg(Xb, w, b), yb).sum()
            loss.backward()
            sgd([w, b], lr, batch_size)
            epoch_loss += loss.item() * len(Xb)
        losses.append(epoch_loss / len(features))
        print(f"[from scratch] epoch {epoch + 1}: loss={losses[-1]:.6f}")
    return w, b, losses


def train_concise(features, labels, batch_size=32, lr=0.03, epochs=5):
    """torch.nn 简洁版：同样的训练循环，用模块和优化器。"""
    net = nn.Sequential(nn.Linear(features.shape[1], 1))
    loss_fn = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    losses = []
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for Xb, yb in loader:
            trainer.zero_grad()
            l = loss_fn(net(Xb), yb)
            l.backward()
            trainer.step()
            epoch_loss += l.item() * len(Xb)
        losses.append(epoch_loss / len(features))
        print(f"[torch.nn] epoch {epoch + 1}: loss={losses[-1]:.6f}")
    return net, losses


def main():
    torch.manual_seed(0)
    features, labels = synthetic_data(TRUE_W, TRUE_B)
    gradient_check(features, labels)

    w, b, scratch_losses = train_scratch(features, labels)
    print("\n从零实现学到的参数:")
    print("  w:", w.flatten().tolist(), "（真实", TRUE_W.tolist(), "）")
    print("  b:", b.item(), "（真实", TRUE_B, "）")

    net, concise_losses = train_concise(features, labels)
    wc, bc = net[0].weight.detach().flatten(), net[0].bias.detach().item()
    print("\ntorch.nn 学到的参数:")
    print("  w:", wc.tolist(), "| b:", round(bc, 3))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(scratch_losses) + 1), scratch_losses, marker="o", label="from scratch")
    ax.plot(range(1, len(concise_losses) + 1), concise_losses, marker="s", label="torch.nn")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Linear regression: loss curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "linear_regression_loss.png"
    fig.savefig(out, dpi=150)
    print("\n已保存:", out)


if __name__ == "__main__":
    main()

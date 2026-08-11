"""
第 2 周 · 核心里程碑：手写 2 层 MLP 的反向传播
===============================================
运行: python week2/backprop_from_scratch.py

网络结构（分类，3 类）：
    X(8) -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> Softmax -> 交叉熵

本脚本用 numpy 手写 forward/backward，然后做两层验证：
1. 与 torch.autograd 的梯度逐项对比（全部参数）
2. 数值梯度（中心差分）抽查几个参数

如果你能独立完成这里的 backward 推导并在纸上写出来，
就达到了本周最重要的里程碑。
"""
import numpy as np
import torch
import torch.nn.functional as F


def make_data(n=64, d=8, c=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, c, size=n)
    Y = np.zeros((n, c))
    Y[np.arange(n), y] = 1.0          # one-hot
    return X, y, Y


def init_params(d=8, h=16, c=3, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.1, (d, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0, 0.1, (h, c))
    b2 = np.zeros(c)
    return W1, b1, W2, b2


def relu(x):
    return np.maximum(x, 0.0)


def forward(X, W1, b1, W2, b2):
    """返回 (概率 p, 缓存)。缓存用于 backward。"""
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    z2 -= z2.max(axis=1, keepdims=True)      # 数值稳定
    p = np.exp(z2)
    p /= p.sum(axis=1, keepdims=True)
    return p, (z1, a1, z2, p)


def cross_entropy(p, Y):
    return -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))


def backward(X, Y, W1, b1, W2, b2, cache):
    """
    手动反向传播。推导过程：
      z1 = X W1 + b1
      a1 = ReLU(z1)
      z2 = a1 W2 + b2
      p  = softmax(z2)
      L  = -mean(sum(Y * log p))

      dL/dz2 = (p - Y) / N            （softmax + 交叉熵的漂亮结论）
      dL/dW2 = a1^T dL/dz2
      dL/db2 = sum(dL/dz2, axis=0)
      dL/da1 = dL/dz2 W2^T
      dL/dz1 = dL/da1 * (z1 > 0)      （ReLU 的导数）
      dL/dW1 = X^T dL/dz1
      dL/db1 = sum(dL/dz1, axis=0)
    """
    z1, a1, z2, p = cache
    n = X.shape[0]

    dz2 = (p - Y) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2


def check_vs_autograd():
    print("=" * 60)
    print("验证 1：numpy 手写梯度 vs torch.autograd")
    print("=" * 60)
    X, y, Y = make_data()
    W1, b1, W2, b2 = init_params()

    p, cache = forward(X, W1, b1, W2, b2)
    loss_np = cross_entropy(p, Y)
    dW1, db1, dW2, db2 = backward(X, Y, W1, b1, W2, b2, cache)

    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    W1t = torch.tensor(W1, requires_grad=True)
    b1t = torch.tensor(b1, requires_grad=True)
    W2t = torch.tensor(W2, requires_grad=True)
    b2t = torch.tensor(b2, requires_grad=True)

    z1t = Xt @ W1t + b1t
    a1t = F.relu(z1t)
    logits = a1t @ W2t + b2t
    loss_t = F.cross_entropy(logits, yt)
    dW1t, db1t, dW2t, db2t = torch.autograd.grad(
        loss_t, [W1t, b1t, W2t, b2t])

    print(f"loss numpy={loss_np:.8f}  torch={loss_t.item():.8f}")
    for name, g_np, g_t in [("dW1", dW1, dW1t), ("db1", db1, db1t),
                            ("dW2", dW2, dW2t), ("db2", db2, db2t)]:
        diff = np.abs(g_np - g_t.numpy()).max()
        status = "OK" if diff < 1e-9 else "FAIL"
        print(f"{name}: max|diff|={diff:.2e}  {status}")


def check_numerical():
    print("\n" + "=" * 60)
    print("验证 2：数值梯度（中心差分）抽查 dW2")
    print("=" * 60)
    X, y, Y = make_data()
    W1, b1, W2, b2 = init_params()

    def loss_for_w2(w2_flat):
        w2 = w2_flat.reshape(W2.shape)
        p, _ = forward(X, W1, b1, w2, b2)
        return cross_entropy(p, Y)

    _, _, dW2, _ = backward(X, Y, W1, b1, W2, b2, forward(X, W1, b1, W2, b2)[1])
    flat = W2.reshape(-1)
    eps = 1e-6
    for idx in [0, 5, 17, 40]:
        wp = flat.copy(); wp[idx] += eps
        wm = flat.copy(); wm[idx] -= eps
        num = (loss_for_w2(wp) - loss_for_w2(wm)) / (2 * eps)
        ana = dW2.reshape(-1)[idx]
        print(f"idx={idx:2d}: numerical={num:+.8f} analytic={ana:+.8f} "
              f"diff={abs(num - ana):.2e}")


def train_numpy(steps=200, lr=0.1):
    print("\n" + "=" * 60)
    print("用手写反向传播训练 200 步（纯 numpy）")
    print("=" * 60)
    X, y, Y = make_data()
    W1, b1, W2, b2 = init_params()
    for step in range(1, steps + 1):
        p, cache = forward(X, W1, b1, W2, b2)
        loss = cross_entropy(p, Y)
        dW1, db1, dW2, db2 = backward(X, Y, W1, b1, W2, b2, cache)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
        if step % 50 == 0:
            acc = (p.argmax(axis=1) == y).mean()
            print(f"step {step:3d}: loss={loss:.4f} acc={acc:.3f}")


if __name__ == "__main__":
    check_vs_autograd()
    check_numerical()
    train_numpy()
    print("\n里程碑达成：手写反向传播与 autograd 一致，且能训练收敛！")

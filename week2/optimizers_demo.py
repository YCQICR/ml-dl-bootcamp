"""
第 2 周 · 练习 6：优化器与正则化对比
=====================================
运行: python week2/optimizers_demo.py --quick --epochs 2

对比：SGD / SGD+Momentum / Adam / Adam+L2 / Adam+Dropout。
注意：不同优化器用不同学习率是"公平比较"的难点，这里按常见默认值设置，
重点看趋势，而不是排名。
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


def make_net(dropout=0.0):
    layers = [nn.Flatten(), nn.Linear(784, 128), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(128, 10))
    return nn.Sequential(*layers)


def run_config(train_loader, test_loader, cfg, epochs, device):
    torch.manual_seed(0)
    net = make_net(cfg["dropout"]).to(device)
    if cfg["opt"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg["lr"],
                                    momentum=cfg["momentum"],
                                    weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    val_accs = []
    for _ in range(epochs):
        net.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(Xb), yb)
            loss.backward()
            optimizer.step()
        val_accs.append(evaluate(net, test_loader, device))
    return val_accs


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            correct += (model(Xb).argmax(1) == yb).sum().item()
            total += len(yb)
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true", help="只用 5000 个样本")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quick = 5000 if args.quick else None
    train_loader, test_loader, *_ = data_utils.load_fashion_mnist(
        batch_size=args.batch_size, quick=quick, synthetic=args.synthetic)

    configs = [
        {"name": "SGD lr=0.01",        "opt": "sgd", "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0, "dropout": 0.0},
        {"name": "SGD+Momentum 0.9",   "opt": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0, "dropout": 0.0},
        {"name": "Adam lr=1e-3",       "opt": "adam", "lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0, "dropout": 0.0},
        {"name": "Adam+L2 5e-4",       "opt": "adam", "lr": 1e-3, "momentum": 0.0, "weight_decay": 5e-4, "dropout": 0.0},
        {"name": "Adam+Dropout 0.2",   "opt": "adam", "lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0, "dropout": 0.2},
    ]

    results = {}
    for cfg in configs:
        print(f"训练: {cfg['name']} ...")
        results[cfg["name"]] = run_config(train_loader, test_loader, cfg, args.epochs, device)

    print("\n结果汇总（每 epoch 的测试集精度）:")
    header = " | ".join(f"{name[:12]:>12}" for name in results)
    print(f"{'epoch':>6} | {header}")
    for e in range(args.epochs):
        row = " | ".join(f"{results[n][e]:>12.4f}" for n in results)
        print(f"{e + 1:>6} | {row}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = range(1, args.epochs + 1)
    for name, accs in results.items():
        ax.plot(epochs, accs, marker="o", label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("Optimizer & regularization comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "optimizers_compare.png"
    fig.savefig(out, dpi=150)
    print("\n已保存:", out)


if __name__ == "__main__":
    main()

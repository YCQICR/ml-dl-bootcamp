"""
第 3 周 · 练习 1：LeNet-5 训练 CIFAR-10
========================================
运行:
    python week3/lenet_cifar10.py --quick --synthetic --epochs 1   # 快速验证
    python week3/lenet_cifar10.py --epochs 20                      # 正式训练

特点：从零写 LeNet 网络结构，带数据增强开关、早停、最佳模型保存、曲线图。
"""
import argparse
import sys
import time
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

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


class LeNet(nn.Module):
    """LeNet-5 的现代版本，适配 3 通道 32x32 输入。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")
    parser.add_argument("--no-augment", action="store_true", help="关闭数据增强")
    parser.add_argument("--quick", action="store_true", help="只用 5000 个样本")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tb", action="store_true", help="开启 TensorBoard")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device)
    quick = 5000 if args.quick else None
    train_loader, test_loader, *_ = data_utils.load_cifar10(
        batch_size=args.batch_size, augment=not args.no_augment,
        quick=quick, synthetic=args.synthetic)

    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = None
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(REPO_ROOT / "runs" / "lenet_cifar10"))

    best_acc, best_state, bad_epochs = 0.0, None, 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        train_acc = correct / total
        val_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(total_loss / total)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:2d}/{args.epochs} | loss={history['train_loss'][-1]:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | {time.time() - t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if writer is not None:
            writer.add_scalar("train/loss", history["train_loss"][-1], epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
        if bad_epochs >= args.patience:
            print(f"早停：连续 {args.patience} 个 epoch 没有提升。")
            break

    ckpt_path = ckpt_dir / "lenet_cifar10_best.pt"
    torch.save({"state_dict": best_state, "best_acc": best_acc}, ckpt_path)
    print(f"最佳验证精度: {best_acc:.4f}，已保存: {ckpt_path}")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"测试集精度: {test_acc:.4f}（目标 > 0.70）")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set_title("train loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="s", label="val")
    axes[1].set_title("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "lenet_cifar10_curves.png"
    fig.savefig(out, dpi=150)
    print("已保存:", out)


if __name__ == "__main__":
    main()

"""
第 2 周 · 练习 5：MLP 训练 Fashion-MNIST（目标 > 85%）
=======================================================
运行:
    python week2/mlp_fashion_mnist.py --quick --epochs 5    # 快速验证
    python week2/mlp_fashion_mnist.py --epochs 10           # 正式训练

亮点：Dropout、Adam、学习率衰减、保存最佳模型、TensorBoard、曲线图。
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


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, num_classes=10, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true", help="只用 10000 个样本")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tb", action="store_true", help="开启 TensorBoard 记录")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device)
    quick = 10000 if args.quick else None
    train_loader, test_loader, *_ = data_utils.load_fashion_mnist(
        batch_size=args.batch_size, quick=quick, synthetic=args.synthetic)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    writer = None
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(REPO_ROOT / "runs" / "mlp_fashion_mnist"))

    best_acc, best_state = 0.0, None
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
        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:2d}/{args.epochs} | loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)

    if best_state is not None:
        ckpt_path = ckpt_dir / "mlp_fashion_mnist_best.pt"
        torch.save({"state_dict": best_state, "best_acc": best_acc}, ckpt_path)
        print(f"\n最佳验证精度: {best_acc:.4f}，已保存: {ckpt_path}")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"测试集精度: {test_acc:.4f}（目标 > 0.85）")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = range(1, args.epochs + 1)
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
    out = FIG_DIR / "mlp_fashion_mnist_curves.png"
    fig.savefig(out, dpi=150)
    print("已保存:", out)


if __name__ == "__main__":
    main()

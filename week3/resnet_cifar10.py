"""
第 3 周 · 练习 2：ResNet-18 训练 CIFAR-10（本周主力项目）
==========================================================
运行:
    python week3/resnet_cifar10.py --quick --synthetic --epochs 1   # 快速验证
    python week3/resnet_cifar10.py --epochs 30                      # 正式训练（带增强）
    python week3/resnet_cifar10.py --epochs 30 --no-augment         # 实验 1：关增强
    python week3/resnet_cifar10.py --epochs 30 --lr 0.01            # 实验 2：换学习率

特点：标准 ResNet-18/34、SGD+Momentum、余弦退火、早停、最佳模型保存。
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def make_resnet(depth):
    if depth == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2])
    if depth == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3])
    raise ValueError("只支持 depth=18 或 34")


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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=18, choices=[18, 34])
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--no-augment", action="store_true", help="关闭数据增强")
    parser.add_argument("--quick", action="store_true", help="只用 5000 个样本")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tb", action="store_true", help="开启 TensorBoard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device} | 模型: ResNet-{args.depth} | 增强: {not args.no_augment} "
          f"| lr={args.lr} | epochs={args.epochs}")

    quick = 5000 if args.quick else None
    train_loader, test_loader, *_ = data_utils.load_cifar10(
        batch_size=args.batch_size, augment=not args.no_augment,
        quick=quick, synthetic=args.synthetic)

    model = make_resnet(args.depth).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = None
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        tag = f"resnet{args.depth}_cifar10_lr{args.lr}_aug{int(not args.no_augment)}"
        writer = SummaryWriter(str(REPO_ROOT / "runs" / tag))

    best_acc, best_state, bad_epochs = 0.0, None, 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
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

        train_acc = correct / total
        val_acc = evaluate(model, test_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(total_loss / total)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        print(f"epoch {epoch:2d}/{args.epochs} | loss={history['train_loss'][-1]:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={current_lr:.2e} "
              f"| {time.time() - t0:.1f}s")

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
            writer.add_scalar("lr", current_lr, epoch)
        if bad_epochs >= args.patience:
            print(f"早停：连续 {args.patience} 个 epoch 没有提升。")
            break

    ckpt_path = ckpt_dir / f"resnet{args.depth}_cifar10_best.pt"
    torch.save({"state_dict": best_state, "best_acc": best_acc,
                "args": vars(args)}, ckpt_path)
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
    out = FIG_DIR / f"resnet{args.depth}_cifar10_curves.png"
    fig.savefig(out, dpi=150)
    print("已保存:", out)


if __name__ == "__main__":
    main()

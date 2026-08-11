"""第 2 周共享数据工具：加载 Fashion-MNIST，无网络/无 torchvision 时退化为合成数据。"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"


def load_fashion_mnist(batch_size=256, quick=None, synthetic=False,
                       num_workers=0, pin_memory=False):
    """
    返回 (train_loader, test_loader, channels, height, width, num_classes)。
    quick=整数时只用前 N 个训练样本（快速验证用）。
    """
    if not synthetic:
        try:
            from torchvision import datasets, transforms
            root = str(DATA_ROOT / "fashion_mnist")
            train_set = datasets.FashionMNIST(
                root=root, train=True, download=True, transform=transforms.ToTensor())
            test_set = datasets.FashionMNIST(
                root=root, train=False, download=True, transform=transforms.ToTensor())
            if quick is not None:
                train_set = Subset(train_set, range(min(quick, len(train_set))))
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory)
            return train_loader, test_loader, 1, 28, 28, 10
        except Exception as exc:
            print(f"[data_utils] Fashion-MNIST 加载失败（{exc}），改用合成数据。")

    n_train = quick if quick else 60000
    n_test = 10000
    train_x = torch.randn(n_train, 1, 28, 28)
    train_y = torch.randint(0, 10, (n_train,))
    test_x = torch.randn(n_test, 1, 28, 28)
    test_y = torch.randint(0, 10, (n_test,))
    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 1, 28, 28, 10

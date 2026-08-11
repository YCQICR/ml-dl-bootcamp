"""第 3 周共享数据工具：加载 CIFAR-10，可选数据增强；无网络/无 torchvision 时退化为合成数据。"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(batch_size=128, augment=False, quick=None, synthetic=False,
                 num_workers=0, pin_memory=False):
    """
    返回 (train_loader, test_loader, channels, height, width, num_classes)。
    augment=True 时对训练集做 RandomCrop + RandomHorizontalFlip。
    """
    if not synthetic:
        try:
            from torchvision import datasets, transforms
            train_tfs = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            if augment:
                train_tfs = [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip()] + train_tfs
            test_tfs = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            root = str(DATA_ROOT / "cifar10")
            train_set = datasets.CIFAR10(root=root, train=True, download=True,
                                         transform=transforms.Compose(train_tfs))
            test_set = datasets.CIFAR10(root=root, train=False, download=True,
                                        transform=transforms.Compose(test_tfs))
            if quick is not None:
                train_set = Subset(train_set, range(min(quick, len(train_set))))
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory)
            return train_loader, test_loader, 3, 32, 32, 10
        except Exception as exc:
            print(f"[data_utils] CIFAR-10 加载失败（{exc}），改用合成数据。")

    n_train = quick if quick else 5000
    n_test = 1000
    train_x = torch.randn(n_train, 3, 32, 32)
    train_y = torch.randint(0, 10, (n_train,))
    test_x = torch.randn(n_test, 3, 32, 32)
    test_y = torch.randint(0, 10, (n_test,))
    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, 3, 32, 32, 10

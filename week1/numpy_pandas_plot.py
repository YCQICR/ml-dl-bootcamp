"""
第 1 周 · 练习 1：Numpy / Pandas / Matplotlib 热身
====================================================
运行:  python week1/numpy_pandas_plot.py
产出:  week1/figures/training_curve_demo.png

目标：
1. Numpy    —— 数组创建、形状变换、广播、索引、矩阵运算
2. Pandas   —— DataFrame 统计、缺失值、分组
3. Matplotlib—— 画一条"训练/验证损失曲线"

学习建议：先整段运行，再逐段修改参数（比如把广播的 10 改成别的数），
然后回答每个 section 末尾的思考题。
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # 不弹窗，直接保存图片（Windows 上更稳定）
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def numpy_section():
    print("=" * 60)
    print("1) Numpy：数组、广播、索引、矩阵运算")
    print("=" * 60)

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print("a =\n", a, "\nshape:", a.shape)

    b = np.arange(6).reshape(2, 3)
    print("b =\n", b)

    # 广播：形状不同的数组也能按规则相加
    print("a + 10（标量广播）=\n", a + 10)

    # 逐元素乘法 vs 矩阵乘法
    print("a * b（逐元素）=\n", a * b)
    print("a @ b.T（矩阵乘法）=\n", a @ b.T)

    # 索引
    print("a[0, :] 第一行:", a[0, :])
    print("a[:, 1] 第二列:", a[:, 1])
    print("a[a > 3] 布尔索引:", a[a > 3])

    # 随机数与统计
    rng = np.random.default_rng(0)
    c = rng.normal(loc=0.0, scale=1.0, size=(1000, 4))
    print("随机矩阵 shape:", c.shape)
    print("整体 mean≈%.3f  std≈%.3f（应接近 0 和 1）" % (c.mean(), c.std()))
    print("每列均值:", c.mean(axis=0).round(3))
    print("前两列的矩阵乘法形状:", (c[:, :2] @ c[:, 2:].T).shape)

    print("\n思考题：a + 10 是怎么做到的？把 10 换成 np.array([1,2,3]) 会发生什么？\n")


def pandas_section():
    print("=" * 60)
    print("2) Pandas：DataFrame、缺失值、分组")
    print("=" * 60)

    df = pd.DataFrame({
        "name":  ["alice", "bob", "carol", "dave"],
        "score": [88, 92, np.nan, 79],
        "hours": [2, 3, 4, 1],
    })
    print(df)
    print("\ndescribe() 统计:\n", df.describe())
    print("\nscore 缺失值数量:", df["score"].isna().sum())

    # 缺失值处理：先用中位数/均值填充（真实项目里要说明理由）
    df["score"] = df["score"].fillna(df["score"].mean())
    print("填充后:\n", df)

    df["grade"] = np.where(df["score"] >= 85, "A", "B")
    print("\n按 grade 分组统计 hours 均值:\n", df.groupby("grade")["hours"].mean())

    out_csv = Path(__file__).resolve().parent / "example_data.csv"
    df.to_csv(out_csv, index=False)
    print("\n已保存示例 CSV:", out_csv)

    print("\n思考题：为什么真实项目里用中位数/均值填充要谨慎？（提示：泄漏）\n")


def plot_section():
    print("=" * 60)
    print("3) Matplotlib：画训练/验证损失曲线")
    print("=" * 60)

    rng = np.random.default_rng(42)
    epochs = np.arange(1, 51)
    # 模拟：训练 loss 持续下降；验证 loss 先降后升（过拟合）
    train_loss = 2.0 * np.exp(-epochs / 12) + 0.05 * rng.normal(size=epochs.size)
    val_loss = 2.0 * np.exp(-epochs / 12) + 0.10 + 0.06 * rng.normal(size=epochs.size)
    val_loss[25:] += 0.02 * (epochs[25:] - 25)   # 后期验证损失回升

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train loss", linewidth=2)
    ax.plot(epochs, val_loss, label="val loss", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Demo: overfitting pattern")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_png = FIG_DIR / "training_curve_demo.png"
    fig.savefig(out_png, dpi=150)
    print("已保存:", out_png)
    print("提问：val loss 后期回升说明什么？→ 过拟合（见 ml_concepts_cheatsheet.md）\n")


if __name__ == "__main__":
    numpy_section()
    pandas_section()
    plot_section()
    print("全部完成！把每节结尾的思考题答案写进你的学习笔记。")

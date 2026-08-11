"""
第 1 周 · 练习 2：一个完整的 sklearn 小项目
============================================
运行:  python week1/titanic_ml.py
产出:  week1/figures/titanic_best_model.png

默认使用 sklearn 内置的糖尿病数据集（离线可跑，回归任务）。
如果你把 Kaggle Titanic 的 train.csv 放到 data/titanic/train.csv，
脚本会自动切换为"是否存活"分类任务。

演示的完整流程（科研/工程都适用）：
1. 数据划分：train / val / test 三份
2. 只用训练集 fit 标准化器（防止数据泄漏）
3. 5 折交叉验证 + 在验证集上选模型
4. 用测试集给出最终指标
5. 画图帮助解释结果
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TITANIC_CSV = BASE.parent / "data" / "titanic" / "train.csv"


def load_titanic(path: Path):
    """读取 Titanic 数据，做最小预处理，返回 (X, y, task)。"""
    df = pd.read_csv(path)
    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[cols + ["Survived"]].copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Sex"] = (df["Sex"] == "male").astype(int)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    X = df.drop(columns=["Survived"]).to_numpy(dtype=float)
    y = df["Survived"].to_numpy()
    return X, y, "classification"


def load_diabetes_data():
    """内置回归数据集，保证离线也能跑通流程。"""
    X, y = load_diabetes(return_X_y=True)
    return X, y, "regression"


def evaluate_classifier(model, Xva, yva):
    p = model.predict(Xva)
    proba = model.predict_proba(Xva)[:, 1]
    return {
        "accuracy": accuracy_score(yva, p),
        "f1": f1_score(yva, p),
        "roc_auc": roc_auc_score(yva, proba),
    }


def evaluate_regressor(model, Xva, yva):
    p = model.predict(Xva)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yva, p))),
        "mae": mean_absolute_error(yva, p),
        "r2": r2_score(yva, p),
    }


def plot_classification(model, Xte, yte):
    proba = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={roc_auc_score(yte, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Titanic: best model ROC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "titanic_best_model.png"
    fig.savefig(out, dpi=150)
    print("已保存:", out)


def plot_regression(model, Xte, yte):
    pred = model.predict(Xte)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(yte, pred, s=18, alpha=0.6)
    lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    ax.plot(lim, lim, "--", color="gray", label="perfect")
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(f"Diabetes: predictions (R2={r2_score(yte, pred):.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "titanic_best_model.png"
    fig.savefig(out, dpi=150)
    print("已保存:", out)


def main():
    if TITANIC_CSV.exists():
        X, y, task = load_titanic(TITANIC_CSV)
        print("使用数据: Titanic（分类任务：是否存活）")
    else:
        X, y, task = load_diabetes_data()
        print("使用数据: sklearn 糖尿病数据集（回归任务）")
        print("提示: 把 Titanic train.csv 放到 data/titanic/ 即可切换为分类任务")
    print("X:", X.shape, "| y:", y.shape, "| 任务:", task)

    # 1) 先分训练+临时，再把临时分成 val / test
    stratify = y if task == "classification" else None
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify)
    stratify2 = ytmp if task == "classification" else None
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, random_state=42, stratify=stratify2)
    print(f"train={len(ytr)}  val={len(yva)}  test={len(yte)}")

    if task == "classification":
        models = {
            "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        }
        metric_names = ["accuracy", "f1", "roc_auc"]
        cv_metric = "accuracy"
    else:
        models = {
            "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        }
        metric_names = ["rmse", "mae", "r2"]
        cv_metric = "r2"

    # 2) 每个模型：fit 训练集 → 验证集打分 → 5 折交叉验证
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        if task == "classification":
            val_scores = evaluate_classifier(model, Xva, yva)
        else:
            val_scores = evaluate_regressor(model, Xva, yva)
        cv = cross_val_score(model, Xtr, ytr, cv=5, scoring=cv_metric)
        results[name] = {"val": val_scores, "cv_mean": cv.mean(), "cv_std": cv.std()}
        print(f"[{name}] val={ {k: round(v, 4) for k, v in val_scores.items()} } "
              f"| 5折CV({cv_metric})={cv.mean():.4f}±{cv.std():.4f}")

    # 3) 在验证集上选最好模型（回归看 R2，分类看 accuracy）
    best_key = "r2" if task == "regression" else "accuracy"
    best_name = max(results, key=lambda n: results[n]["val"][best_key])
    best_model = models[best_name]
    print(f"\n验证集上最好的模型: {best_name}")

    # 4) 最后用测试集给一次最终指标
    if task == "classification":
        test_scores = evaluate_classifier(best_model, Xte, yte)
        plot_classification(best_model, Xte, yte)
    else:
        test_scores = evaluate_regressor(best_model, Xte, yte)
        plot_regression(best_model, Xte, yte)
    print("测试集最终指标:", {k: round(v, 4) for k, v in test_scores.items()})

    print("\n总结问题：")
    print("1. 交叉验证的平均值和验证集分数一致吗？差别说明什么？")
    print("2. 换成另一个模型/超参数，测试集分数会怎么变？")
    print("3. 如果反复用测试集挑模型，会发生什么？")


if __name__ == "__main__":
    main()

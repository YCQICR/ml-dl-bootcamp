# 第 1 周：工具 + 机器学习直觉

> 预计投入：20–25 小时（环境 8h + 课程视频 10h + sklearn 项目 5h）
> 验收标准：能解释过拟合与偏差-方差；能读懂一条训练/验证曲线的含义。

## 本周任务

- [ ] 按 `setup_guide.md` 完成环境搭建，并验证 GPU 可用
- [ ] 看吴恩达《机器学习》课程（1.5 倍速）以下小节：
  - 线性回归与代价函数
  - 梯度下降
  - 逻辑回归与分类
  - 过拟合、正则化
  - 模型评估（偏差-方差、学习曲线）
- [ ] 运行 `numpy_pandas_plot.py`，逐段看懂并亲手改参数
- [ ] 运行 `titanic_ml.py`，理解 train/val/test 与交叉验证
- [ ] 阅读 `ml_concepts_cheatsheet.md` 并用自己的话复述一遍

## 运行方式

```powershell
cd outputs\ml-dl-bootcamp
conda activate ml-dl
python week1\numpy_pandas_plot.py
python week1\titanic_ml.py
```

## 本周代码

| 文件 | 作用 |
|---|---|
| `setup_guide.md` | 环境安装（Conda、PyTorch、Git、Jupyter） |
| `numpy_pandas_plot.py` | Numpy/Pandas/Matplotlib 热身练习 |
| `titanic_ml.py` | 完整 sklearn 小项目：划分、交叉验证、评估、画图 |
| `ml_concepts_cheatsheet.md` | 机器学习核心概念速查卡 |

## 检查自己是否真的懂了

1. 什么是过拟合？训练误差和验证误差分别会怎么变？
2. 为什么必须把测试集留到最后？
3. 交叉验证解决了什么问题？
4. 数据标准化为什么只能"fit"在训练集上？

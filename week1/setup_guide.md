# 环境搭建指南（Windows）

本指南按"你已经有一台 Windows 电脑 + 一块 NVIDIA 显卡（RTX 3060 6GB 已实测可用）"编写。

## 1. 认识你的 Python

你的电脑上已经安装了 Conda（位置：`D:\conda`，Python 3.13.9），它暂时没有加入 PATH，
所以直接在 PowerShell 里输入 `python` 会提示找不到命令。解决办法有两种，任选其一：

**方案 A（推荐）：初始化 conda，让 `conda activate` 直接可用**

在 PowerShell 里执行一次：

```powershell
D:\conda\Scripts\conda.exe init powershell
```

然后**关掉并重新打开 PowerShell**。之后输入 `conda activate ml-dl` 即可进入课程环境。

> 注意：如果提示"禁止运行脚本"，先执行：
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> 再重新打开 PowerShell。

**方案 B：不修改 PATH，每次用完整路径**

```powershell
conda run -n ml-dl python week1\numpy_pandas_plot.py
```

或直接运行环境内的 Python：

```powershell
D:\conda\envs\ml-dl\python.exe week1\numpy_pandas_plot.py
```

## 2. 课程环境（已由 Codex 创建好）

课程专用环境名为 `ml-dl`（Python 3.11），位于 `D:\conda\envs\ml-dl`。
里面已经安装了 PyTorch（CUDA 版）、torchvision、TensorBoard、JupyterLab 以及
numpy/pandas/matplotlib/scikit-learn 等。

你的显卡驱动版本是 551.52（支持 CUDA 12.4），所以 PyTorch 选用了匹配的
**torch 2.6.0+cu124** 版本。

验证环境：

```powershell
conda activate ml-dl
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果输出类似 `2.6.0+cu124 True`，说明 GPU 可用。

### 如果以后想重装或补装依赖

```powershell
conda activate ml-dl
pip install -r requirements.txt
```

万一需要重装 CUDA 版 PyTorch（与当前驱动匹配）：

```powershell
conda activate ml-dl
pip install --force-reinstall torch==2.6.0+cu124 torchvision==0.21.0+cu124 `
    --index-url https://download.pytorch.org/whl/cu124
```

> 提示：以后如果更新了 NVIDIA 显卡驱动（建议更新到 570+），就可以换用更新的
> cu126/cu128 版本；具体命令去 https://pytorch.org/get-started/locally/ 查询。

## 3. Jupyter Notebook

```powershell
conda activate ml-dl
jupyter lab
```

浏览器会自动打开 JupyterLab。本课程代码以 `.py` 脚本为主，方便直接运行；
你想边看边改的话，也可以在 Jupyter 里打开 `.py` 文件逐段执行。

## 4. Git 与 GitHub

### 4.1 配置身份（只做一次）

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

### 4.2 把本课程仓库变成自己的 GitHub 仓库

1. 在 GitHub 网页上新建一个空仓库（不要勾选 README）。
2. 在项目目录执行：

```powershell
cd outputs\ml-dl-bootcamp
git init
git add .
git commit -m "一个月 ML/DL 快速入门：第 1 周"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

3. 以后每次学完一批内容：

```powershell
git add .
git commit -m "第 X 周：完成……"
git push
```

## 5. 常见问题

| 问题 | 解决办法 |
|---|---|
| `python` 不是内部或外部命令 | 先 `conda activate ml-dl`，或用 `D:\conda\envs\ml-dl\python.exe` |
| 下载 PyTorch 很慢 | 已使用清华镜像；还慢就换网或挂代理 |
| `torch.cuda.is_available()` 返回 False | 运行 `nvidia-smi` 看驱动版本，再按上面命令重装对应 cu1xx 版 PyTorch |
| 训练时显存不够 | 调小 `--batch-size`（如 64），本机 6GB 显存跑 CIFAR-10 没问题 |
| matplotlib 中文乱码 | 本课程图表统一用英文标签，避免字体问题 |

## 6. 之后每个脚本怎么用

```powershell
python week2\mlp_fashion_mnist.py --help    # 查看所有参数
python week2\mlp_fashion_mnist.py --quick   # 快速模式（少数据，验证流程）
python week2\mlp_fashion_mnist.py --synthetic # 无网络时用随机数据
```

先用 `--quick` 跑通流程，再正式训练。

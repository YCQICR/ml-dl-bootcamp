# Transformer 背景笔记（本月只要求"看懂概念"）

配合阅读 <https://jalammar.github.io/illustrated-transformer/>。
目标：知道 Transformer 是什么、解决什么问题，不做代码实现（下个月做）。

## 为什么需要 Transformer

- RNN 按时间步处理序列，无法并行，长距离依赖容易丢失。
- 注意力机制（Attention）可以让模型"直接看到"序列中任意位置的信息。
- Transformer = 纯注意力 + 前馈网络，抛弃循环结构，可并行训练。

## 核心概念

### 1. Self-Attention（自注意力）

- 每个词向量生成三个向量：Query（查询）、Key（键）、Value（值）。
- 用 Query 和所有 Key 算相似度（点积），softmax 后作为权重，
  加权求和所有 Value 得到输出。
- 直觉：每个词"询问"其他词与自己有多相关，再按相关度汇总信息。

### 2. Scaled Dot-Product Attention

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

除以 sqrt(d_k) 防止点积过大导致 softmax 梯度消失。

### 3. Multi-Head Attention（多头注意力）

- 把 Q/K/V 分成多个头，各自学不同角度的关系（语法、指代、语义……），最后拼接。
- 一个头只能学到一种"相关性"，多头让模型同时关注多种关系。

### 4. Positional Encoding（位置编码）

- 注意力本身不区分顺序（"我爱猫"和"猫爱我"的每个词都一样相关）。
- 给每个位置加一个正弦/余弦编码，让模型知道词的先后。

### 5. Encoder-Decoder 结构

- Encoder：把输入序列编码成上下文表示（自注意力 + 前馈）。
- Decoder：逐个生成输出（带掩码的自注意力 + 对 Encoder 的交叉注意力）。
- BERT 只用 Encoder（理解任务），GPT 只用 Decoder（生成任务）。

## 与课程的关系

- 第 2 周学的 MLP、第 3 周学的 CNN，都是"局部/固定结构"的特征提取；
  Transformer 用注意力做"全局动态"的特征交互。
- 现代大模型（LLM）的核心就是 Decoder-only Transformer + 大规模预训练。

## 本月检查问题

1. 自注意力输出怎么计算？Q、K、V 各是什么？
2. 为什么需要多头和位置编码？
3. BERT 和 GPT 在结构上的区别是什么？

下个月任务预告：用 d2l 实现一个迷你 Transformer，再用 Hugging Face 微调一个预训练模型。

# 笔记

## Tokenizer 改造 —— Adapter-based Character-aware Model

**核心痛点**：目前的 LLM（如 LLaMA, Qwen）使用的是 BPE 分词，通过 SentencePiece 将常用词（如“篮球”）切成一个 Token。这导致模型无法看到“篮”和“球”内部的字形结构，也难以处理“篮”被写成“蓝”这种**字级别**的错误。

### 1. 模型设计 (The Architecture)

不重新预训练整个 LLM（成本太高），而是设计一个 **"Plug-in Character Adapter"**。

- **Step 1: 强制单字切分 (Forced Char Tokenization)**
  - 在输入端，强制将中文句子按“字”切分（例如：`["我", "爱", "打", "蓝", "球"]`），而不是 BPE 的子词。
- **Step 2: 词表对齐 (Vocabulary Alignment)**
  - LLM 的词表里通常包含常用的单字。如果包含，直接调用其 Embedding；如果不包含，使用 `<UNK>` 或扩展词表。
- **Step 3: Adapter 注入 (The Key Innovation)**
  - 在 Embedding Layer 之后，Transformer Layers 之前，插入一个 **"Char-Context Adapter"**。
  - **结构**：一个轻量级的 Transformer Encoder 或 MLP。
  - **功能**：由于强制单字切分破坏了 LLM 原本习惯的 BPE 语义组合模式，Adapter 的作用是将“单字 Embedding 序列”映射回 LLM 熟悉的“语义空间”，同时注入拼音/字形特征。
  - **输入增强**：$Input_{adapter} = E_{char} + E_{pinyin} + E_{glyph}$。
- **训练策略**：冻结 LLM 参数，只微调 Adapter 和 LLM 的输出头（Prediction Head）。

> **参考文献支撑**：
>
> - **思路来源**：*C-LLM (2024)* 证明了 Character-level Tokenization 的必要性。
> - **技术手段**：参考 *ABC-Fusion (2024)* 在 BERT 上使用 Adapter 的方式，将其迁移到 LLM 上。

### 2. 实验推进

- **主实验**：在 SIGHAN 数据集上，对比 **Original LLM (BPE)** vs **Ours (Char-Adapter)**。预期你的模型在“形近字”错误上的召回率（Recall）会显著提升。
- **分析实验**：
  - **OOV (Out-of-Vocabulary) 测试**：构建包含生僻字或领域术语的测试集（如医疗、法律），证明 Char-level Adapter 比 BPE 更具鲁棒性。

---
title: "InvestLMOptions<T>"
description: "Configuration options for InvestLM (Investment Language Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for InvestLM (Investment Language Model).

## For Beginners

InvestLM focuses on investment applications:

**The Key Insight:**
Investment decisions require understanding of market dynamics, company fundamentals,
and economic indicators. InvestLM is trained to process investment-relevant text
and provide insights for portfolio management.

**What Problems Does InvestLM Solve?**

- Stock recommendation reasoning
- Portfolio analysis and suggestions
- Market research summarization
- Investment thesis generation
- Risk factor analysis
- Earnings call insights extraction

**Key Benefits:**

- Understands investment terminology
- Can reason about market conditions
- Trained on investment research corpus
- Supports various investment workflows

## How It Works

InvestLM is a large language model specifically designed for investment-related
NLP tasks including portfolio analysis, stock recommendation, and market research.

**Reference:** Yang et al., "InvestLM: A Large Language Model for Investment", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InvestLMOptions` | Initializes a new instance with default InvestLM configuration. |
| `InvestLMOptions(InvestLMOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate (default: 0.1). |
| `HiddenDimension` | Hidden dimension (default: 768). |
| `IntermediateDimension` | Intermediate feed-forward dimension (default: 3072). |
| `MaxSequenceLength` | Maximum sequence length in tokens (default: 2048). |
| `NumAttentionHeads` | Number of attention heads (default: 12). |
| `NumClasses` | Number of output classes (default: 3 for recommendation: buy/hold/sell). |
| `NumLayers` | Number of transformer layers (default: 12). |
| `TaskType` | Task type: "recommendation", "sentiment", "qa", "summary" (default: "recommendation"). |
| `VocabularySize` | Vocabulary size (default: 32000 for LLaMA-style). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the InvestLM options. |


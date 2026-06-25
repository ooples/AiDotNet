---
title: "FinancialBERTOptions<T>"
description: "Configuration options for FinancialBERT model - a domain-adapted BERT for comprehensive financial analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FinancialBERT model - a domain-adapted BERT for comprehensive financial analysis.

## For Beginners

FinancialBERT provides comprehensive financial NLP:

**The Key Insight:**
While FinBERT focuses on sentiment and SEC-BERT on regulatory filings, FinancialBERT
provides broader coverage of the financial domain including analyst reports, market
commentary, corporate communications, and financial news.

**What Problems Does FinancialBERT Solve?**

- Multi-task financial text classification
- Extracting insights from analyst reports
- Processing corporate press releases
- Analyzing market commentary and research notes
- Understanding financial terminology across contexts

**Key Benefits:**

- Broader domain coverage than specialized models
- Multi-task learning capability
- Effective for general financial NLP applications
- Good baseline for financial text understanding

## How It Works

FinancialBERT is a BERT model adapted for broad financial domain understanding,
including news analysis, market commentary, analyst reports, and general financial text.

**Reference:** Huang et al., "FinancialBERT: A Pre-trained Language Model for Financial Text Mining", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialBERTOptions` | Initializes a new instance with default FinancialBERT configuration. |
| `FinancialBERTOptions(FinancialBERTOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate (default: 0.1). |
| `HiddenDimension` | Hidden dimension (default: 768 for BERT-base). |
| `IntermediateDimension` | Intermediate feed-forward dimension (default: 3072). |
| `MaxSequenceLength` | Maximum sequence length in tokens (default: 512). |
| `NumAttentionHeads` | Number of attention heads (default: 12). |
| `NumClasses` | Number of output classes (default: 3 for sentiment). |
| `NumLayers` | Number of transformer layers (default: 12). |
| `TaskType` | Task type: "sentiment", "topic", "entity", "multi" (default: "sentiment"). |
| `UseMultiTask` | Whether to use multi-task learning (default: false). |
| `VocabularySize` | Vocabulary size (default: 30522 for BERT-base). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the FinancialBERT options. |


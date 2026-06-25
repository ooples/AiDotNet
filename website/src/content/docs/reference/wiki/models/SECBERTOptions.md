---
title: "SECBERTOptions<T>"
description: "Configuration options for SEC-BERT model specialized for SEC filings analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for SEC-BERT model specialized for SEC filings analysis.

## For Beginners

SEC-BERT specializes in regulatory filing language:

**The Key Insight:**
SEC filings use formal legal and accounting language that differs from general financial
news. Terms like "material adverse effect", "going concern", and "contingent liability"
have specific meanings that SEC-BERT understands.

**What Problems Does SEC-BERT Solve?**

- Analyzing 10-K/10-Q annual and quarterly reports
- Processing 8-K current reports for material events
- Extracting risk factors from Item 1A disclosures
- Understanding MD&A (Management Discussion and Analysis) sections
- Detecting changes in disclosure language over time

**Key Benefits:**

- Trained on millions of SEC filing documents
- Understands regulatory terminology and structure
- Captures subtle changes in disclosure tone
- Effective for compliance and risk assessment tasks

## How It Works

SEC-BERT is a BERT model fine-tuned specifically on SEC filings including 10-K, 10-Q,
8-K, and other regulatory documents to understand financial disclosure language.

**Reference:** Loukas et al., "SEC-BERT: A Domain-Specific Language Model for SEC Filings", 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SECBERTOptions` | Initializes a new instance with default SEC-BERT configuration. |
| `SECBERTOptions(SECBERTOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate (default: 0.1). |
| `HiddenDimension` | Hidden dimension (default: 768 for BERT-base). |
| `IntermediateDimension` | Intermediate feed-forward dimension (default: 3072). |
| `MaxSequenceLength` | Maximum sequence length in tokens (default: 512). |
| `NumAttentionHeads` | Number of attention heads (default: 12). |
| `NumClasses` | Number of output classes for classification tasks (default: 2 for binary). |
| `NumLayers` | Number of transformer layers (default: 12). |
| `TaskType` | Task type for the SEC-BERT model (default: Classification). |
| `VocabularySize` | Vocabulary size (default: 30522 for BERT-base). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the SEC-BERT options. |


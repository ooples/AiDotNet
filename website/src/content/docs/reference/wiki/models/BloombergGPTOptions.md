---
title: "BloombergGPTOptions<T>"
description: "Configuration options for BloombergGPT-style financial language model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for BloombergGPT-style financial language model.

## For Beginners

BloombergGPT represents finance-specialized LLMs:

**The Key Insight:**
General LLMs (GPT-3, GPT-4) lack deep financial knowledge. BloombergGPT-style models
are trained on massive financial corpora to understand terminology, concepts, and
relationships specific to finance.

**What Problems Does BloombergGPT Solve?**

- Financial sentiment analysis with high accuracy
- Named entity recognition for financial entities
- Financial question answering
- News classification and headline generation
- Understanding complex financial documents

**Architecture Highlights:**

- Large-scale decoder-only transformer
- Trained on mixed financial and general text
- Supports various financial NLP benchmarks

**Key Benefits:**

- Deep financial domain knowledge
- State-of-the-art on financial NLP tasks
- Understands complex financial terminology
- Can handle nuanced financial queries

## How It Works

BloombergGPT-style models are large language models trained on extensive financial
data including Bloomberg's proprietary financial text corpus along with general text.

**Reference:** Wu et al., "BloombergGPT: A Large Language Model for Finance", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BloombergGPTOptions` | Initializes a new instance with default configuration. |
| `BloombergGPTOptions(BloombergGPTOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate (default: 0.1). |
| `HiddenDimension` | Hidden dimension (default: 1024 for medium model). |
| `IntermediateDimension` | Intermediate feed-forward dimension (default: 4096). |
| `MaxSequenceLength` | Maximum sequence length in tokens (default: 2048). |
| `NumAttentionHeads` | Number of attention heads (default: 16). |
| `NumClasses` | Number of output classes (default: 3). |
| `NumLayers` | Number of transformer layers (default: 24). |
| `TaskType` | Task type: "classification", "ner", "qa", "generation" (default: "classification"). |
| `VocabularySize` | Vocabulary size (default: 50257). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the BloombergGPT options. |


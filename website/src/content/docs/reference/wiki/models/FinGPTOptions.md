---
title: "FinGPTOptions<T>"
description: "Configuration options for FinGPT (Financial GPT) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FinGPT (Financial GPT) model.

## For Beginners

FinGPT brings GPT capabilities to finance:

**The Key Insight:**
While BERT models excel at understanding tasks (classification, NER), GPT models
are better at generation tasks and can handle more open-ended financial queries.
FinGPT adapts the GPT architecture for financial applications.

**What Problems Does FinGPT Solve?**

- Financial sentiment analysis with explanations
- Question answering about financial documents
- Generating financial summaries
- Extracting insights from earnings calls
- Financial text completion and generation

**Key Benefits:**

- Open-source and fine-tunable
- Supports both understanding and generation
- Can provide reasoning for predictions
- Handles longer context than BERT models

## How It Works

FinGPT is an open-source financial large language model designed for various
financial NLP tasks including sentiment analysis, question answering, and text generation.

**Reference:** Yang et al., "FinGPT: Open-Source Financial Large Language Models", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinGPTOptions` | Initializes a new instance with default FinGPT configuration. |
| `FinGPTOptions(FinGPTOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate (default: 0.1). |
| `HiddenDimension` | Hidden dimension (default: 768 for GPT-2 small). |
| `IntermediateDimension` | Intermediate feed-forward dimension (default: 3072). |
| `MaxSequenceLength` | Maximum sequence length in tokens (default: 2048 for GPT-style). |
| `NumAttentionHeads` | Number of attention heads (default: 12). |
| `NumClasses` | Number of output classes for classification tasks (default: 3). |
| `NumLayers` | Number of transformer layers (default: 12). |
| `TaskType` | Task type: "classification", "generation", "qa" (default: "classification"). |
| `VocabularySize` | Vocabulary size (default: 50257 for GPT-2 tokenizer). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the FinGPT options. |


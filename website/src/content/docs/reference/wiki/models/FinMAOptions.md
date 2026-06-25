---
title: "FinMAOptions<T>"
description: "Configuration options for FinMA (Financial Multi-Agent) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FinMA (Financial Multi-Agent) model.

## For Beginners

FinMA uses multi-agent approach for financial NLP:

**The Key Insight:**
Complex financial tasks often require multiple capabilities (sentiment analysis,
entity extraction, numerical reasoning). FinMA coordinates specialized agents
to handle different aspects of a financial analysis task.

**What Problems Does FinMA Solve?**

- Multi-faceted financial document analysis
- Complex financial question answering
- Coordinated analysis of earnings reports
- Integrated sentiment and entity extraction
- Financial reasoning with multiple information sources

**Key Benefits:**

- Specialized agents for different tasks
- Better handling of complex queries
- Modular and extensible architecture
- Can combine multiple analysis types

## How It Works

FinMA is a financial multi-agent LLM system designed to handle complex financial
tasks through specialized agent coordination.

**Reference:** Zhang et al., "FinMA: A Multi-Agent Financial LLM System", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinMAOptions` | Initializes a new instance with default FinMA configuration. |
| `FinMAOptions(FinMAOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate (default: 0.1). |
| `HiddenDimension` | Hidden dimension (default: 768). |
| `IntermediateDimension` | Intermediate feed-forward dimension (default: 3072). |
| `MaxSequenceLength` | Maximum sequence length in tokens (default: 2048). |
| `NumAgents` | Number of specialized agents (default: 4). |
| `NumAttentionHeads` | Number of attention heads (default: 12). |
| `NumClasses` | Number of output classes (default: 3). |
| `NumLayers` | Number of transformer layers (default: 12). |
| `VocabularySize` | Vocabulary size (default: 32000 for LLaMA-style). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the FinMA options. |


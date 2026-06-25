---
title: "FinBERTToneOptions<T>"
description: "Configuration options for FinBERT-Tone model for fine-grained financial sentiment/tone analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FinBERT-Tone model for fine-grained financial sentiment/tone analysis.

## For Beginners

FinBERT-Tone provides nuanced sentiment analysis:

**The Key Insight:**
Standard sentiment models use 3 classes (negative/neutral/positive), but financial
communications often have more subtle tones. FinBERT-Tone uses 5 classes to capture
gradations like "cautiously optimistic" vs "strongly bullish".

**What Problems Does FinBERT-Tone Solve?**

- Earnings call tone analysis (detecting management confidence levels)
- Forward-looking statement sentiment classification
- Analyst report tone assessment
- Corporate communication sentiment tracking
- Detecting subtle changes in disclosure tone over time

**The 5 Tone Classes:**

1. Very Negative: Strong pessimism, significant concerns
2. Negative: Pessimistic outlook, concerns mentioned
3. Neutral: Factual, no clear sentiment
4. Positive: Optimistic outlook, confidence expressed
5. Very Positive: Strong optimism, high confidence

**Key Benefits:**

- Fine-grained sentiment detection
- Better captures management tone nuances
- Useful for sentiment momentum tracking
- Effective for earnings call analysis

## How It Works

FinBERT-Tone is a FinBERT variant specifically focused on capturing fine-grained
sentiment and tone in financial communications. It uses 5 classes to capture more
nuanced sentiment than standard 3-class models.

**Reference:** Huang et al., "FinBERT: A Pretrained Language Model for Financial Communications", 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinBERTToneOptions` | Initializes a new instance with default FinBERT-Tone configuration. |
| `FinBERTToneOptions(FinBERTToneOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Dropout rate (default: 0.1). |
| `HiddenDimension` | Hidden dimension (default: 768 for BERT-base). |
| `IntermediateDimension` | Intermediate feed-forward dimension (default: 3072). |
| `MaxSequenceLength` | Maximum sequence length in tokens (default: 512). |
| `NumAttentionHeads` | Number of attention heads (default: 12). |
| `NumLayers` | Number of transformer layers (default: 12). |
| `NumToneClasses` | Number of tone classes (default: 5 for fine-grained sentiment). |
| `UseFinegrainedTone` | Whether to use fine-grained 5-class tone (true) or standard 3-class sentiment (false). |
| `VocabularySize` | Vocabulary size (default: 30522 for BERT-base). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the FinBERT-Tone options. |


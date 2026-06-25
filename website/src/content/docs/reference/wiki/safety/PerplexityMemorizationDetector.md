---
title: "PerplexityMemorizationDetector<T>"
description: "Detects potential training data memorization by estimating text perplexity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects potential training data memorization by estimating text perplexity.
Low perplexity (highly predictable) text may indicate verbatim memorized content.

## For Beginners

If an AI can predict every next word perfectly, it's probably
reciting something it memorized during training. Normal text has some unpredictability.
This module measures how predictable the text is — too predictable means it might be
a memorized passage from training data.

## How It Works

Memorized training data typically has unusually low perplexity — the model is very
"confident" about every next token because it has seen the exact sequence before.
This detector computes a lightweight character-level perplexity proxy using n-gram
statistics and flags outputs that are suspiciously predictable compared to natural text.

**Detection approach:**

1. Build character-level n-gram frequency table from the text itself
2. Compute conditional entropy at each position using n-gram context
3. Average entropy = proxy for perplexity
4. Very low entropy indicates memorized/formulaic text
5. Additional checks: token repetition patterns, entropy variance

**References:**

- Detecting pre-training data via perplexity comparison (Carlini et al., 2023)
- Min-K%/Min-K%++ membership inference (2024, arxiv:2404.02936)
- BookMIA: Book-level membership inference (2024, arxiv:2401.15588)
- Scalable extraction of training data from LLMs (2023, arxiv:2311.17035)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerplexityMemorizationDetector(Double,Double,Int32)` | Initializes a new perplexity-based memorization detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |


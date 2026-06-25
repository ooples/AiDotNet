---
title: "EqualizedOddsChecker<T>"
description: "Checks for equalized odds violations by analyzing whether model quality or effort varies across demographic groups mentioned in the text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Fairness`

Checks for equalized odds violations by analyzing whether model quality or effort
varies across demographic groups mentioned in the text.

## For Beginners

Imagine asking an AI to write a recommendation letter for two
equally qualified candidates. If the letter for a man is detailed and enthusiastic but
the letter for a woman is shorter and uses more hedging words like "might" or "could",
that's an equalized odds violation — the model is giving different quality output for
different groups.

## How It Works

Equalized odds requires that the true positive rate and false positive rate are equal
across all demographic groups. Since we're working with text (not tabular classification),
this module detects differential quality indicators: response length disparity, detail
level differences, hedging language, and conditional/qualifying statements that differ
by demographic group.

**References:**

- Equalized Odds in Machine Learning (Hardt et al., NeurIPS 2016)
- Measuring algorithmic fairness in text generation (2024)
- BEATS: Comprehensive bias evaluation test suite for LLMs (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EqualizedOddsChecker(Double)` | Initializes a new equalized odds checker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `EvaluateText(String)` |  |


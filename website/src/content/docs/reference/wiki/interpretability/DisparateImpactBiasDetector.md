---
title: "DisparateImpactBiasDetector<T>"
description: "Detects bias using the Disparate Impact metric (80% rule)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Detects bias using the Disparate Impact metric (80% rule).
Disparate Impact Ratio = (Min Positive Rate) / (Max Positive Rate).
A ratio below 0.8 indicates potential bias.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DisparateImpactBiasDetector(Double)` | Initializes a new instance of the DisparateImpactBiasDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBiasDetectionResult(Vector<>,Vector<>,Vector<>)` | Implements bias detection using Disparate Impact ratio (80% rule). |


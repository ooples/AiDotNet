---
title: "EqualOpportunityBiasDetector<T>"
description: "Detects bias using Equal Opportunity metric (True Positive Rate difference)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Detects bias using Equal Opportunity metric (True Positive Rate difference).
Requires actual labels to compute TPR for each group.
A TPR difference greater than 0.1 (10%) indicates potential bias.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EqualOpportunityBiasDetector(Double)` | Initializes a new instance of the EqualOpportunityBiasDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBiasDetectionResult(Vector<>,Vector<>,Vector<>)` | Implements bias detection using Equal Opportunity (TPR difference). |


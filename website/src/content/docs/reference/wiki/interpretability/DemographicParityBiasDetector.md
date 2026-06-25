---
title: "DemographicParityBiasDetector<T>"
description: "Detects bias using Demographic Parity (Statistical Parity Difference)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability`

Detects bias using Demographic Parity (Statistical Parity Difference).
Measures the difference in positive prediction rates between groups.
A difference greater than 0.1 (10%) indicates potential bias.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DemographicParityBiasDetector(Double)` | Initializes a new instance of the DemographicParityBiasDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBiasDetectionResult(Vector<>,Vector<>,Vector<>)` | Implements bias detection using Statistical Parity Difference. |


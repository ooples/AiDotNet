---
title: "FairnessConfig"
description: "Configuration for fairness and bias detection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for fairness and bias detection.

## For Beginners

Fairness settings check whether the model treats all demographic
groups equitably. For example, ensuring the model doesn't give different quality
results for different genders or ethnicities.

## How It Works

**References:**

- BEATS: Comprehensive bias evaluation test suite for LLMs (2025)
- SB-Bench: Stereotype bias benchmark for multimodal models (2025)
- Demographic-targeted bias: race/ethnicity 55.6% exploitability (2025)

## Properties

| Property | Summary |
|:-----|:--------|
| `DemographicParity` | Gets or sets whether demographic parity checking is enabled. |
| `EqualizedOdds` | Gets or sets whether equalized odds checking is enabled. |
| `ProtectedAttributes` | Gets or sets the protected attributes to monitor. |
| `StereotypeDetection` | Gets or sets whether stereotype detection is enabled. |


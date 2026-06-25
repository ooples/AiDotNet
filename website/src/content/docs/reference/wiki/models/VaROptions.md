---
title: "VaROptions<T>"
description: "Configuration options for Value-at-Risk (VaR) models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Value-at-Risk (VaR) models.

## For Beginners

VaR options define how conservative your risk estimate is
and how much data the model expects. Think of them as the "risk dial" and the
"input size" settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceLevel` | The confidence level for VaR calculation (e.g., 0.95 or 0.99). |
| `NumFeatures` | Number of input features used for risk calculation. |
| `TimeHorizon` | The time horizon for the risk assessment (in days). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the VaR options. |


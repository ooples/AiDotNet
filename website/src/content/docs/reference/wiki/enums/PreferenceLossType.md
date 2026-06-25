---
title: "PreferenceLossType"
description: "Types of loss functions for preference optimization methods."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Types of loss functions for preference optimization methods.

## For Beginners

This controls how the model learns from preference data.
Start with Sigmoid (standard DPO) and only switch if you have specific needs.

## How It Works

Different preference optimization methods use different loss formulations.
Each has tradeoffs in terms of stability, sample efficiency, and robustness.

## Fields

| Field | Summary |
|:-----|:--------|
| `Conservative` | Conservative DPO loss (cDPO). |
| `Hinge` | Hinge loss for preference optimization. |
| `IPO` | Identity Preference Optimization (IPO) loss. |
| `KTO` | Kahneman-Tversky Optimization loss. |
| `OddsRatio` | Odds ratio preference loss (used in ORPO). |
| `Robust` | Robust DPO loss with outlier handling. |
| `Sigmoid` | Standard sigmoid loss used in DPO. |
| `Simple` | Simple preference optimization loss (used in SimPO). |


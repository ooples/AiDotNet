---
title: "GaussianDifferentialPrivacyVector<T>"
description: "Implements Gaussian differential privacy for vector-based model updates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy`

Implements Gaussian differential privacy for vector-based model updates.

## How It Works

**For Beginners:** This adds carefully calibrated random noise to a parameter vector so that
individual data points cannot be inferred from the update, while the overall signal remains useful.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetClipNorm` | Gets the gradient clipping norm used for sensitivity bounding. |
| `ResetPrivacyBudget` | Resets the privacy budget counter. |


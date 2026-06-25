---
title: "FewTUREUncertaintyMethod"
description: "Uncertainty estimation method for FewTURE."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Uncertainty estimation method for FewTURE.

## Fields

| Field | Summary |
|:-----|:--------|
| `Entropy` | Prediction entropy: H(p) = -sum(p_i * log(p_i)) |
| `MCDropout` | Monte Carlo dropout-based uncertainty. |
| `Variance` | Prediction variance across token-level features. |


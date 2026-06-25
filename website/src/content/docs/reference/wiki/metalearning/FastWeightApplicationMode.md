---
title: "FastWeightApplicationMode"
description: "Specifies how fast weights are applied to modify the base model."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Specifies how fast weights are applied to modify the base model.

## Fields

| Field | Summary |
|:-----|:--------|
| `Additive` | Fast weights are added to base weights: θ' = θ + α |
| `FiLM` | Feature-wise Linear Modulation: output' = γ × output + β |
| `Multiplicative` | Fast weights scale base weights: θ' = θ × (1 + α) |


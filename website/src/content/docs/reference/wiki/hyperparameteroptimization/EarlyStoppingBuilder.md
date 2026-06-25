---
title: "EarlyStoppingBuilder<T>"
description: "Builder for configuring early stopping with fluent API."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.HyperparameterOptimization`

Builder for configuring early stopping with fluent API.

## Methods

| Method | Summary |
|:-----|:--------|
| `Build` | Builds the EarlyStopping instance. |
| `Create` | Creates a new builder for configuring early stopping. |
| `Maximize` | Configures for maximization (higher is better). |
| `Minimize` | Configures for minimization (lower is better). |
| `WithMinDelta(Double)` | Sets the minimum improvement delta. |
| `WithMode(EarlyStoppingMode)` | Sets the improvement mode. |
| `WithPatience(Int32)` | Sets the patience (number of non-improving checks before stopping). |


---
title: "RWKVForecastingOptions<T>"
description: "Configuration options for RWKV-based time series forecasting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for RWKV-based time series forecasting.

## For Beginners

RWKV combines the best of both worlds:

**Key Properties:**

1. **Linear Complexity:** O(n) for training and inference (vs O(n^2) for Transformers)
2. **Constant Memory:** O(1) per-token generation memory
3. **Parallel Training:** Can be computed as a convolution for efficient parallel training
4. **Multi-head:** Multiple attention heads for better capacity

**Architecture:**

- Time mixing: WKV attention mechanism with learned decay
- Channel mixing: FFN with gating
- Residual connections and layer normalization

## How It Works

RWKV (Receptance Weighted Key Value) is a linear-complexity sequence model that combines
the efficient training parallelism of Transformers with the constant-memory inference of RNNs.
This options class configures an RWKV model for time series forecasting tasks.

**Reference:** Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RWKVForecastingOptions` | Initializes a new instance with default values. |
| `RWKVForecastingOptions(RWKVForecastingOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `ModelDimension` | Gets or sets the model dimension (d_model). |
| `NumHeads` | Gets or sets the number of RWKV heads. |
| `NumLayers` | Gets or sets the number of RWKV layers. |


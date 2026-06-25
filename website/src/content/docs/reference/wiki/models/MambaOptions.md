---
title: "MambaOptions<T>"
description: "Configuration options for Mamba (Selective State Space Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Mamba (Selective State Space Model).

## For Beginners

Mamba is a breakthrough in efficient sequence modeling:

**The Key Insight:**
Transformers have O(n^2) complexity due to attention, which is slow for long sequences.
State space models (SSMs) have O(n) complexity but are less expressive.
Mamba makes SSM parameters input-dependent (selective), combining the best of both.

**How It Works:**

1. **State Space Model:** Maintains a hidden state updated recurrently
2. **Selective Mechanism:** Parameters (A, B, C, delta) vary with input
3. **Hardware-aware Algorithm:** Efficient implementation via parallel scan
4. **Linear Complexity:** O(n) time and memory for sequence length n

**Architecture:**

- Input projection to expanded dimension
- 1D convolution for local context
- Selective SSM core with input-dependent parameters
- Output projection back to model dimension

**Advantages:**

- Linear time complexity (vs O(n^2) for attention)
- Handles very long sequences efficiently
- Strong performance on language, audio, and time series
- Hardware-efficient implementation

## How It Works

Mamba is a selective state space model that achieves linear-time complexity for
sequence modeling while maintaining the expressiveness of transformers through
input-dependent (selective) state space parameters.

**Reference:** Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
https://arxiv.org/abs/2312.00752

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MambaOptions` | Initializes a new instance of the `MambaOptions` class with default values. |
| `MambaOptions(MambaOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `ConvKernelSize` | Gets or sets the convolution kernel size. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `DtRank` | Gets or sets the rank for the delta (dt) projection. |
| `ExpandFactor` | Gets or sets the expansion factor for the inner dimension. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `ModelDimension` | Gets or sets the model dimension (d_model). |
| `NumLayers` | Gets or sets the number of Mamba layers. |
| `StateDimension` | Gets or sets the state dimension (d_state or N). |
| `UseBidirectional` | Gets or sets whether to use bidirectional processing. |


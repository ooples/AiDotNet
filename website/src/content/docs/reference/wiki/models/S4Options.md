---
title: "S4Options<T>"
description: "Configuration options for S4 (Structured State Space Sequence Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for S4 (Structured State Space Sequence Model).

## For Beginners

S4 is a breakthrough model that showed state space models
can match transformers on long-range sequence tasks:

**The Key Insight:**
State space models have linear complexity O(n), but historically couldn't compete with
attention. S4 solves this by:

1. Using HiPPO (High-order Polynomial Projection Operators) for state initialization
2. Structuring the state matrix A as a diagonal plus low-rank (DPLR) matrix
3. Computing convolutions efficiently using FFT

**How It Works:**

1. **HiPPO Matrix:** Initializes A to optimally compress history into the state
2. **DPLR Decomposition:** A = diagonal + low-rank for efficient computation
3. **Discretization:** Converts continuous SSM to discrete for sequence data
4. **FFT Convolution:** Computes SSM as convolution using O(n log n) FFT

**The Math (simplified):**

- State update: x' = Ax + Bu (A is HiPPO-structured)
- Output: y = Cx + Du
- Discretized: x_k = A_bar * x_{k-1} + B_bar * u_k
- For long sequences: compute as convolution K * u using FFT

**Advantages:**

- Near-linear complexity O(n log n) via FFT
- Excellent on Long Range Arena benchmark
- Handles sequences up to 16K+ tokens
- Foundation for Mamba and other modern SSMs

## How It Works

S4 is a foundational state space model that achieves near-linear complexity through
structured parameterization of the state transition matrix using the HiPPO framework.

**Reference:** Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", 2022.
https://arxiv.org/abs/2111.00396

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `S4Options` | Initializes a new instance of the `S4Options` class with default values. |
| `S4Options(S4Options<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DiscretizationMethod` | Gets or sets the discretization method for converting continuous to discrete SSM. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HippoMethod` | Gets or sets the HiPPO method for state matrix initialization. |
| `LowRankRank` | Gets or sets the rank of the low-rank correction. |
| `ModelDimension` | Gets or sets the model dimension (d_model). |
| `NumLayers` | Gets or sets the number of S4 layers. |
| `StateDimension` | Gets or sets the state dimension (N in the paper). |
| `UseBidirectional` | Gets or sets whether to use bidirectional processing. |
| `UseLowRankCorrection` | Gets or sets whether to use low-rank correction in the DPLR decomposition. |


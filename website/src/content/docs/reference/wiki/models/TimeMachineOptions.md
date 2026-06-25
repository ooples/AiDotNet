---
title: "TimeMachineOptions<T>"
description: "Configuration options for TimeMachine (Time Series State Space Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TimeMachine (Time Series State Space Model).

## For Beginners

TimeMachine is a modern architecture that combines ideas from
state space models (like Mamba and S4) with time series-specific enhancements:

**The Key Insight:**
While Mamba and S4 are general-purpose sequence models, TimeMachine is specifically
designed for time series data with features like:

1. Multi-scale temporal decomposition
2. Trend-seasonality modeling
3. Efficient long-range dependency capture

**How It Works:**

1. **Temporal Decomposition:** Separates trend, seasonal, and residual components
2. **Multi-Scale SSM:** Processes different temporal scales with dedicated SSM blocks
3. **Adaptive Gating:** Learns which scales are most important for each prediction
4. **Reconstruction:** Combines multi-scale outputs for final forecast

**Architecture:**

- Input embedding with reversible instance normalization
- Multi-scale SSM blocks (fine, medium, coarse granularity)
- Scale-wise attention for importance weighting
- Output projection with de-normalization

**Advantages:**

- Linear complexity O(n) from SSM backbone
- Explicit temporal decomposition improves interpretability
- Multi-scale processing captures patterns at different frequencies
- State-of-the-art results on time series benchmarks

## How It Works

TimeMachine is a state space model specifically designed for time series forecasting
that combines the efficiency of SSMs with specialized temporal modeling components.

**Reference:** Ahamed et al., "TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting", 2024.
https://arxiv.org/abs/2403.09898

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeMachineOptions` | Initializes a new instance of the `TimeMachineOptions` class with default values. |
| `TimeMachineOptions(TimeMachineOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `ConvKernelSize` | Gets or sets the convolution kernel size for local context. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ExpandFactor` | Gets or sets the expansion factor for SSM inner dimension. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `ModelDimension` | Gets or sets the model dimension (d_model). |
| `NumLayers` | Gets or sets the number of SSM layers per scale. |
| `NumScales` | Gets or sets the number of temporal scales to model. |
| `StateDimension` | Gets or sets the state dimension for each SSM block. |
| `TemporalDecompositionMethod` | Gets or sets the temporal decomposition method for multi-scale processing. |
| `UseMultiScaleAttention` | Gets or sets whether to use multi-scale attention for combining scales. |
| `UseReversibleNormalization` | Gets or sets whether to use reversible instance normalization. |


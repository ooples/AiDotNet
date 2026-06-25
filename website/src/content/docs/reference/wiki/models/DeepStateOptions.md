---
title: "DeepStateOptions<T>"
description: "Configuration options for the DeepState (Deep State Space) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the DeepState (Deep State Space) model.

## For Beginners

DeepState is like giving a classical statistics model a brain:

**What is a State Space Model?**
SSMs assume your data has hidden "states" that evolve over time:

- State transition: z_t = F * z_{t-1} + noise (how states evolve)
- Observation: y_t = H * z_t + noise (how states produce observations)

**Example - Trend + Seasonality:**
States might represent:

- Level (current baseline)
- Trend (direction of change)
- Seasonal patterns (weekly/yearly cycles)

The observed value is a combination of these hidden states.

**Why "Deep" State Space?**
Classical SSMs need you to specify the model structure (how many seasonal patterns, etc.).
DeepState uses a neural network to:

- Automatically learn appropriate state representations
- Adapt to complex, non-linear patterns
- Share patterns across multiple time series

**Benefits:**

- Interpretable decomposition (trend, seasonality, residual)
- Natural uncertainty quantification
- Handles multiple related time series well
- Works with irregular data and missing values

## How It Works

DeepState combines deep learning with classical state space models (SSM) for probabilistic
time series forecasting. A neural network learns to parameterize the state space model,
while the SSM structure provides interpretable components like trend and seasonality.

**Reference:** Rangapuram et al., "Deep State Space Models for Time Series Forecasting", 2018.
https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepStateOptions` | Initializes a new instance of the `DeepStateOptions` class with default values. |
| `DeepStateOptions(DeepStateOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension of the RNN encoder. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LookbackWindow` | Gets or sets the lookback window size (input sequence length). |
| `NumRnnLayers` | Gets or sets the number of RNN layers. |
| `SeasonalPeriods` | Gets or sets the seasonal periods to model. |
| `StateDimension` | Gets or sets the state dimension of the state space model. |
| `UseSeasonality` | Gets or sets whether to include seasonality components. |
| `UseTrend` | Gets or sets whether to include a trend component. |


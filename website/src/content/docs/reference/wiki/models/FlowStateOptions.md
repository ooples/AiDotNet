---
title: "FlowStateOptions<T>"
description: "Configuration options for FlowState (IBM's SSM-based Time Series Foundation Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FlowState (IBM's SSM-based Time Series Foundation Model).

## For Beginners

FlowState proves bigger isn't always better:

**State-Space Model Architecture:**
Instead of attention (which is quadratic in sequence length), FlowState uses
structured state spaces (like S4/Mamba) that are linear in sequence length.
This makes it extremely efficient for long sequences.

**Key Advantages:**

- Only 9.1M parameters (smallest in GIFT-Eval top 10)
- Outperforms models 20x its size
- Generalizes to unseen timescales
- Linear-time computation for long sequences

## How It Works

FlowState is IBM's State-Space Model (SSM) based time series foundation model with only
9.1M parameters. Despite being the smallest model in the GIFT-Eval top 10, it outperforms
models 20x its size and generalizes to unseen timescales.

**Reference:** IBM Research, "SSM Time Series Model", 2025.
https://research.ibm.com/blog/SSM-time-series-model

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlowStateOptions` | Initializes a new instance with default values. |
| `FlowStateOptions(FlowStateOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumLayers` | Gets or sets the number of SSM layers. |
| `SSMRank` | Gets or sets the SSM rank for low-rank parameterization. |
| `StateDimension` | Gets or sets the state dimension for the SSM. |
| `UseDiscretization` | Gets or sets whether to use discretization for continuous-time SSM. |


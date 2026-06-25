---
title: "DiffusionTSOptions<T>"
description: "Configuration options for DiffusionTS (Interpretable Diffusion for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for DiffusionTS (Interpretable Diffusion for Time Series).

## For Beginners

DiffusionTS focuses on making diffusion models more
interpretable by decomposing time series into understandable components:

**The Key Insight:**
Time series often have clear structure (trends, seasonality) that gets lost in
"black box" models. DiffusionTS preserves this structure by generating each
component separately and combining them.

**How DiffusionTS Works:**

1. **Decomposition:** Split time series into trend, seasonal, and residual
2. **Component Diffusion:** Generate each component with specialized networks
3. **Reconstruction:** Combine components to form final forecast
4. **Interpretation:** Each component has clear meaning

**DiffusionTS Architecture:**

- Trend Network: Captures long-term movements (slow, smooth)
- Seasonal Network: Captures periodic patterns (daily, weekly, yearly)
- Residual Network: Captures irregular fluctuations
- Fusion Module: Combines components coherently

**Key Benefits:**

- Interpretable decomposition of forecasts
- Can enforce structural constraints (smooth trends, periodic seasons)
- Better uncertainty quantification per component
- Enables "what-if" analysis by modifying components

## How It Works

DiffusionTS is an interpretable diffusion model for time series that uses seasonal-trend
decomposition to generate forecasts with clear interpretable components.

**Reference:** Yuan and Qiu, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", 2024.
https://arxiv.org/abs/2403.01742

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionTSOptions` | Initializes a new instance of the `DiffusionTSOptions` class with default values. |
| `DiffusionTSOptions(DiffusionTSOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BetaEnd` | Gets or sets the ending noise level. |
| `BetaSchedule` | Gets or sets the noise schedule type. |
| `BetaStart` | Gets or sets the starting noise level. |
| `DecompositionPeriod` | Gets or sets the decomposition period for seasonal-trend separation. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the main hidden dimension. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion steps. |
| `NumFeatures` | Gets or sets the number of features. |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `SeasonalHiddenDim` | Gets or sets the hidden dimension for the seasonal network. |
| `SequenceLength` | Gets or sets the sequence length (input length). |
| `TrendHiddenDim` | Gets or sets the hidden dimension for the trend network. |
| `TrendKernelSize` | Gets or sets the kernel size for trend extraction. |
| `UseSeasonalComponent` | Gets or sets whether to model seasonal component. |
| `UseTrendComponent` | Gets or sets whether to model trend component. |


---
title: "TimeGPTOptions<T>"
description: "Configuration options for TimeGPT-style time series forecasting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TimeGPT-style time series forecasting model.

## For Beginners

TimeGPT brings GPT-like capabilities to time series:

**The Key Insight:**
Just as GPT was trained on internet-scale text data to become a general-purpose
language model, TimeGPT is trained on millions of diverse time series to become
a general-purpose forecasting model.

**Core Features:**

1. **Large-scale Pre-training:** Trained on millions of time series
2. **Zero-shot Forecasting:** No training needed for new data
3. **Uncertainty Quantification:** Provides prediction intervals
4. **Multi-horizon:** Forecasts at any horizon

**Architecture:**

- Positional encoding for temporal information
- Multi-head self-attention for pattern recognition
- Large transformer backbone
- Conformal prediction for uncertainty

**Advantages:**

- Works out-of-the-box on new time series
- No hyperparameter tuning required
- Handles various frequencies and domains
- Production-ready forecasting API style

## How It Works

TimeGPT represents a GPT-style architecture adapted for time series forecasting,
featuring large-scale pre-training on diverse time series data with zero-shot
and few-shot forecasting capabilities.

**Reference:** Garza et al., "TimeGPT-1", 2023.
https://arxiv.org/abs/2310.03589

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeGPTOptions` | Initializes a new instance of the `TimeGPTOptions` class with default values. |
| `TimeGPTOptions(TimeGPTOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceLevel` | Gets or sets the confidence level for prediction intervals. |
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `FineTuningLearningRate` | Gets or sets the learning rate for fine-tuning. |
| `FineTuningSteps` | Gets or sets the number of fine-tuning steps for domain adaptation. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension size. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `UseConformalPrediction` | Gets or sets whether to use conformal prediction for uncertainty quantification. |


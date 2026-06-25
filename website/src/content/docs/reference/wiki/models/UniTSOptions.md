---
title: "UniTSOptions<T>"
description: "Configuration options for UniTS (Unified Time Series Model)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for UniTS (Unified Time Series Model).

## For Beginners

UniTS is designed to be a universal time series model:

**The Key Insight:**
Different time series tasks share common patterns. Instead of training
separate models, UniTS learns a unified representation that works for all tasks.

**Supported Tasks:**

1. **Forecasting:** Predict future values
2. **Classification:** Categorize entire time series
3. **Anomaly Detection:** Identify unusual patterns
4. **Imputation:** Fill in missing values

**Architecture:**

- Multi-scale temporal convolution for local patterns
- Transformer layers for global dependencies
- Task-specific heads for different outputs
- Shared backbone pretrained on diverse datasets

**Advantages:**

- One model for multiple tasks (transfer learning)
- Strong zero-shot performance on new domains
- Efficient inference (shared computation)

## How It Works

UniTS is a unified architecture for multiple time series tasks including
forecasting, classification, anomaly detection, and imputation using
a single pretrained model.

**Reference:** Gao et al., "UniTS: A Unified Multi-Task Time Series Model", 2024.
https://arxiv.org/abs/2403.00131

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniTSOptions` | Initializes a new instance of the `UniTSOptions` class with default values. |
| `UniTSOptions(UniTSOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `ConvKernelSizes` | Gets or sets the convolution kernel sizes for multi-scale temporal convolution. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension. |
| `NumClasses` | Gets or sets the number of classes for classification task. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `TaskType` | Gets or sets the task type. |


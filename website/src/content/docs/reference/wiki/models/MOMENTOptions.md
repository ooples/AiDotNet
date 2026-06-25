---
title: "MOMENTOptions<T>"
description: "Configuration options for MOMENT (Multi-task Optimization through Masked Encoding for Time series) foundation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for MOMENT (Multi-task Optimization through Masked Encoding for
Time series) foundation model.

## For Beginners

MOMENT is like a Swiss Army knife for time series:

**Key Innovation — Multi-Task Architecture:**
Unlike most time series models that only do forecasting, MOMENT handles 5 tasks:

1. **Forecasting**: Predict future values
2. **Anomaly Detection**: Find unusual patterns via reconstruction
3. **Classification**: Label time series segments
4. **Imputation**: Fill in missing values
5. **Embedding**: Generate vector representations

**How It Works:**

- Divides input into patches (like Vision Transformer for images)
- Applies RevIN to handle different scales and distributions
- Uses a T5-style transformer encoder to process patches
- Task-specific heads generate outputs for each task type

**Model Sizes:**

- Small (~40M params): Fast experiments
- Base (~385M params): Strong general-purpose performance
- Large (~1B+ params): Maximum capacity

## How It Works

MOMENT is a multi-task time series foundation model from Carnegie Mellon University.
It uses a T5-based encoder-only transformer with patch embeddings and RevIN
(Reversible Instance Normalization) to handle diverse time series tasks including
forecasting, anomaly detection, classification, imputation, and embedding generation.

**Reference:** Goswami et al., "MOMENT: A Family of Open Time-Series Foundation Models",
ICML 2024. https://arxiv.org/abs/2402.03885

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MOMENTOptions` | Initializes a new instance with default values. |
| `MOMENTOptions(MOMENTOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenDimension` | Gets or sets the hidden dimension of the T5 transformer. |
| `IntermediateSize` | Gets or sets the intermediate size for the feed-forward network. |
| `MaskRatio` | Gets or sets the mask ratio for pretraining and imputation tasks. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumClasses` | Gets or sets the number of classification classes (for classification task only). |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer encoder layers. |
| `PatchLength` | Gets or sets the patch length for input tokenization. |
| `Task` | Gets or sets the active task for the model. |


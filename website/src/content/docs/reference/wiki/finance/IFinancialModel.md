---
title: "IFinancialModel<T>"
description: "Base interface for all financial AI models in AiDotNet.Finance."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Base interface for all financial AI models in AiDotNet.Finance.

## For Beginners

Financial AI models in this library follow a dual-mode pattern:

- **Native Mode:** Uses pure C# neural network layers for training and inference.

This mode supports gradient computation, parameter updates, and full training capabilities.
Use this when you need to train models from scratch or fine-tune on your data.

- **ONNX Mode:** Loads pretrained models in ONNX format for fast inference only.

This mode is optimized for production deployment where you don't need training.
Use this when deploying pretrained models for prediction.

All financial models share common capabilities:

- Time series forecasting
- Uncertainty quantification (prediction intervals)
- Financial metrics computation (Sharpe ratio, drawdown, etc.)
- Integration with the AiDotNet ecosystem (serialization, checkpointing, etc.)

## How It Works

This interface extends `IFullModel` with financial-specific capabilities,
including support for both ONNX inference and native trainable implementations.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumFeatures` | Gets the number of input features (variables) the model expects. |
| `PredictionHorizon` | Gets the model's prediction horizon (number of future time steps to forecast). |
| `SequenceLength` | Gets the model's expected input sequence length. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forecast(Tensor<>,Double[])` | Generates forecasts with uncertainty quantification. |
| `GetFinancialMetrics` | Gets financial-specific metrics from the model. |


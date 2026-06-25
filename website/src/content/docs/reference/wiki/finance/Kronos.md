---
title: "Kronos<T>"
description: "Kronos — Foundation Model for the Language of Financial Markets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Kronos — Foundation Model for the Language of Financial Markets.

## For Beginners

Kronos is a foundation model built specifically for financial
markets. It was trained on over 12 billion candlestick records from 45 exchanges worldwide,
so it natively understands the language of stock charts (open, high, low, close, volume).
Think of it as a model that has "read" every trading chart in history and can predict what
comes next based on patterns it has learned.

## How It Works

Kronos is a decoder-only foundation model pre-trained on 12B+ K-line (candlestick) records
across 45 global exchanges. It natively understands OHLCV (Open, High, Low, Close, Volume)
candlestick patterns for financial market forecasting.

**Reference:** "Kronos: A Foundation Model for the Language of Financial Markets", 2025.
https://arxiv.org/abs/2508.02739

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Kronos(NeuralNetworkArchitecture<>,KronosOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Kronos model in native mode for training or fine-tuning. |
| `Kronos(NeuralNetworkArchitecture<>,String,KronosOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Kronos model using a pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `CreateNewInstance` |  |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |


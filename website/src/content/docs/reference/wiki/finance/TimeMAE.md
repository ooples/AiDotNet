---
title: "TimeMAE<T>"
description: "TimeMAE — Masked Autoencoder for Time Series with Decoupled Masked Autoencoders."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TimeMAE — Masked Autoencoder for Time Series with Decoupled Masked Autoencoders.

## For Beginners

TimeMAE learns about time series by hiding random chunks of
data and training a model to fill them back in, similar to how a student learns vocabulary
by doing fill-in-the-blank exercises. The model only processes the visible chunks with a
large encoder and uses a smaller decoder to reconstruct the hidden ones, making training
very efficient. The learned representations transfer well to forecasting tasks.

## How It Works

TimeMAE applies masked autoencoding to time series, randomly masking patches of the input
and training a transformer to reconstruct the missing patches, learning rich temporal representations.
It uses an asymmetric encoder-decoder architecture where the encoder processes visible patches only.

**Reference:** Cheng et al., "TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders", 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeMAE(NeuralNetworkArchitecture<>,String,TimeMAEOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimeMAE model using a pretrained ONNX model. |
| `TimeMAE(NeuralNetworkArchitecture<>,TimeMAEOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TimeMAE model in native mode for training or fine-tuning. |

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
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `PretrainMaskedReconstruction(Tensor<>,Nullable<Int32>)` | Performs one TimeMAE masked-reconstruction pretraining forward pass per Cheng et al. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |


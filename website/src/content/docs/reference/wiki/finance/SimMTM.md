---
title: "SimMTM<T>"
description: "SimMTM — Simple Pre-Training Framework for Masked Time-Series Modeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

SimMTM — Simple Pre-Training Framework for Masked Time-Series Modeling.

## For Beginners

SimMTM learns about time series by playing a fill-in-the-blank
game. Parts of the data are hidden (masked), and the model must reconstruct them. What makes
it special is that it looks at similar series in the training batch for clues, like asking
a friend who has seen similar patterns. This pre-training approach helps the model learn
robust representations that transfer well to forecasting tasks.

## How It Works

SimMTM combines masked time series modeling with series-level similarity learning,
recovering masked series by aggregating from similar unmasked series in the batch.
It uses a patch-based transformer with a similarity-weighted reconstruction objective.

**Reference:** Dong et al., "SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling", NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimMTM(NeuralNetworkArchitecture<>,SimMTMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a SimMTM model in native mode for training or fine-tuning. |
| `SimMTM(NeuralNetworkArchitecture<>,String,SimMTMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a SimMTM model using a pretrained ONNX model. |

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
| `PretrainSimilarityWeightedReconstruction(Tensor<>,Nullable<Int32>)` | Performs one SimMTM similarity-weighted masked-reconstruction pretraining forward pass per Dong et al. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |


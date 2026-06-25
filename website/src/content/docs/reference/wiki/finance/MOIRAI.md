---
title: "MOIRAI<T>"
description: "MOIRAI (Masked EncOder-based UnIveRsAl TIme Series Foundation Model) implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

MOIRAI (Masked EncOder-based UnIveRsAl TIme Series Foundation Model) implementation.

## For Beginners

MOIRAI is designed to be truly universal:

**Multi-Scale Patching:**
Unlike single-patch models, MOIRAI uses multiple patch sizes simultaneously:

- Small patches (8 steps): Capture fine-grained, high-frequency patterns
- Medium patches (16, 32): Balance detail and context
- Large patches (64+): Capture long-term trends and seasonality

**Masked Encoder Architecture:**
During training, random patches are masked and the model learns to predict them.
This is similar to BERT's masked language modeling but for time series.

**Mixture of Distributions:**
For probabilistic forecasting, MOIRAI outputs a mixture of Gaussian distributions,
allowing it to model complex, multi-modal forecast uncertainties.

**Any-to-Any Forecasting:**
The same model can predict any horizon from any context length, making it
flexible for different forecasting scenarios.

## How It Works

MOIRAI is Salesforce's universal time series foundation model that uses multi-scale
patching and masked encoder training for any-to-any forecasting. It can handle
different time series frequencies and domains without fine-tuning.

**Reference:** Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", 2024.
https://arxiv.org/abs/2402.02592

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MOIRAI(NeuralNetworkArchitecture<>,MOIRAIOptions<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance in native mode for training. |
| `MOIRAI(NeuralNetworkArchitecture<>,String,MOIRAIOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance using an ONNX pretrained model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `MaxContextLength` |  |
| `MaxPredictionHorizon` |  |
| `ModelSize` |  |
| `NumFeatures` |  |
| `NumMixtures` | Gets the number of mixture components for distribution output. |
| `PatchSize` |  |
| `PatchSizes` | Gets the array of patch sizes used for multi-scale patching. |
| `PredictionHorizon` |  |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplyRandomMasking(Tensor<>)` | Applies random masking to input for masked encoder training. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors into a single tensor. |
| `ConvertFromOnnxTensor(Tensor<Single>)` | Converts ONNX tensor back to our tensor type. |
| `ConvertToFloatArray(Tensor<>)` | Converts tensor to float array for ONNX compatibility. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads MOIRAI-specific configuration during deserialization. |
| `Dispose(Boolean)` | Releases resources used by the model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExpandToQuantiles(Tensor<>)` | Expands point predictions to quantile forecasts by applying learned quantile offsets. |
| `ExtractLayerReferences` | Extracts references to key layers for efficient access during forward/backward pass. |
| `ExtractMedianFromQuantiles(Tensor<>,Int32)` | Extracts the median (point) forecast from the quantile output tensor. |
| `ExtractPointPredictions(Tensor<>,Int32)` | Extracts point predictions from mixture distribution parameters. |
| `ExtractPointPredictionsTapeSafe(Tensor<>,Int32)` | Tape-safe mixture-weighted-mean extraction of point predictions from a rank-2 mixture-head output. |
| `ExtractRequestedQuantiles(Tensor<>,Int32,Double[])` | Extracts specific quantile levels from the full quantile output. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastOnnx(Tensor<>)` | Performs ONNX-based inference for forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the network. |
| `ForwardDecoderOnly(Tensor<>)` | Decoder-only forward pass for Moirai 2.0. |
| `ForwardDecoderOnlyForTraining(Tensor<>)` | Tape-safe equivalent of `Tensor{` used by `Tensor{` when `_useDecoderOnly` is true. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward pass. |
| `GenerateMixtureQuantiles(Tensor<>,Int32,Double[])` | Generates quantile predictions from mixture distribution. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the model layers based on configuration. |
| `PredictCore(Tensor<>)` |  |
| `SampleStandardNormal(Random)` | Samples from a standard normal distribution using Box-Muller transform. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes MOIRAI-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by appending predictions and removing oldest values. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet MOIRAI requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_attentionLayers` | References to the transformer attention layers. |
| `_contextLength` | Context length for the input sequence. |
| `_dropout` | Dropout rate. |
| `_embeddingLayer` | Reference to the multi-scale embedding layer. |
| `_forecastHorizon` | Forecast horizon for predictions. |
| `_hiddenDimension` | Hidden dimension of the transformer. |
| `_intermediateSize` | Intermediate size for FFN. |
| `_lossFunction` | The loss function used for training. |
| `_maskRatio` | Mask ratio for training. |
| `_modelSize` | Model size variant. |
| `_numFeatures` | Number of input features. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer layers. |
| `_numMixtures` | Number of mixture components. |
| `_optimizer` | The optimizer used for training the model. |
| `_outputHead` | Reference to the distribution output head. |
| `_patchSizes` | Patch sizes for multi-scale patching. |
| `_totalPatches` | Total number of patches across all scales. |
| `_useNativeMode` | Indicates whether the model is running in native mode (true) or ONNX mode (false). |


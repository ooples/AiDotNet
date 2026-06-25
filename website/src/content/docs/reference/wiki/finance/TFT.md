---
title: "TFT<T>"
description: "Temporal Fusion Transformer (TFT) neural network for multi-horizon time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Transformers`

Temporal Fusion Transformer (TFT) neural network for multi-horizon time series forecasting.

## For Beginners

TFT is like a smart assistant that considers different types of information:

- **Static features:** Things that don't change (e.g., store location)
- **Known future inputs:** Things we know ahead (e.g., holidays, promotions)
- **Unknown inputs:** Historical data we can only observe (e.g., past sales)

Key innovations:

- **Variable Selection:** Automatically finds which features matter most
- **Gated Skip Connections:** Helps information flow through the network
- **Interpretable Attention:** Shows which time periods influenced predictions

## How It Works

TFT is a state-of-the-art architecture that combines high-performance multi-horizon forecasting
with interpretable insights. It uses variable selection networks, gating mechanisms, and
self-attention to handle multiple input types (static, known future, unknown observed).

**Reference:** Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon
Time Series Forecasting", International Journal of Forecasting 2021.
https://arxiv.org/abs/1912.09363

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TFT` | Creates a TFT model with default configuration for native training. |
| `TFT(NeuralNetworkArchitecture<>,String,TemporalFusionTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TFT network using pretrained ONNX model. |
| `TFT(NeuralNetworkArchitecture<>,TemporalFusionTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TFT network in native mode for training from scratch. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
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
| `AddGatedConnection(Tensor<>,Tensor<>)` | Adds a gated connection between input and processed output. |
| `AdjustToPredictionHorizon(Tensor<>)` | Adjusts output to match prediction horizon. |
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `ApplyRevIN(Tensor<>,Boolean)` | Applies RevIN normalization/denormalization. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads TFT-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the TFT model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through the TFT network. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for TFT. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes TFT-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input and appends predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet TFT's architectural requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_attentionLayer` | Interpretable multi-head attention layer. |
| `_decoderVariableSelection` | Variable selection network for decoder (future) features. |
| `_dropout` | Dropout rate for regularization. |
| `_encoderVariableSelection` | Variable selection network for encoder (historical) features. |
| `_finalNorm` | Final layer normalization. |
| `_grnLayers` | Gated residual network layers. |
| `_hiddenSize` | Hidden state size for the model. |
| `_instanceMean` | Instance normalization mean (for RevIN). |
| `_instanceStd` | Instance normalization standard deviation (for RevIN). |
| `_lossFunction` | The loss function for training. |
| `_lstmDecoder` | LSTM decoder for processing future sequence. |
| `_lstmEncoder` | LSTM encoder for processing historical sequence. |
| `_numFeatures` | The number of input features. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer/GRN layers. |
| `_optimizer` | The optimizer for training. |
| `_outputProjection` | Output projection layer for quantile predictions. |
| `_predictionHorizon` | The prediction horizon. |
| `_quantileLevels` | Quantile levels for probabilistic forecasting. |
| `_sequenceLength` | The input sequence length (lookback window). |
| `_staticCovariateSize` | Size of static covariate inputs. |
| `_staticVariableSelection` | Variable selection network for static features. |
| `_useInstanceNormalization` | Whether to use instance normalization (RevIN). |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |
| `_useVariableSelection` | Whether to use variable selection networks. |


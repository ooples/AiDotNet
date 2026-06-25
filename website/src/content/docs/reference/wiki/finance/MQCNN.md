---
title: "MQCNN<T>"
description: "MQCNN (Multi-Quantile Convolutional Neural Network) for probabilistic time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

MQCNN (Multi-Quantile Convolutional Neural Network) for probabilistic time series forecasting.

## For Beginners

MQCNN gives you confidence intervals with your predictions:

**What are Quantiles?**
Quantiles are percentiles of the predicted distribution:

- 10th percentile (P10): 10% of values fall below this
- 50th percentile (P50): The median (50% above, 50% below)
- 90th percentile (P90): 90% of values fall below this

**Why Predict Multiple Quantiles?**
Instead of just saying "tomorrow's price will be $100":

- P10: $95 (likely lower bound - 90% chance actual is above this)
- P50: $100 (median prediction)
- P90: $105 (likely upper bound - 90% chance actual is below this)

**The Architecture:**

1. **Encoder:** Dilated convolutions extract temporal patterns from history
2. **Context:** Compressed representation of the encoded sequence
3. **Decoder:** Produces quantile predictions from the context

**Quantile Loss (Pinball Loss):**
Unlike MSE which penalizes all errors equally, quantile loss:

- For P90: Penalizes under-predictions more than over-predictions
- For P10: Penalizes over-predictions more than under-predictions
- For P50: Equal penalty (reduces to MAE)

**Example Use Cases:**

- Stock price prediction with confidence bounds
- Demand forecasting with safety stock levels
- Energy load forecasting with peak/valley estimates

## How It Works

MQCNN is a probabilistic forecasting model that predicts multiple quantiles simultaneously,
providing uncertainty estimates along with point forecasts. It uses an encoder-decoder architecture
with dilated causal convolutions for temporal modeling.

**Reference:** Wen et al., "A Multi-Horizon Quantile Recurrent Forecaster", 2017.
https://arxiv.org/abs/1711.11053

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MQCNN(NeuralNetworkArchitecture<>,MQCNNOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an MQCNN in native mode for training from scratch. |
| `MQCNN(NeuralNetworkArchitecture<>,String,MQCNNOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an MQCNN using pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `NumFeatures` |  |
| `NumQuantiles` | Gets the number of quantiles this model predicts. |
| `PatchSize` |  |
| `PredictionHorizon` |  |
| `Quantiles` | Gets the quantile levels being predicted. |
| `SequenceLength` |  |
| `Stride` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInstanceNormalization(Tensor<>)` |  |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `CalculateCoverage(Tensor<>,Tensor<>,Double,Double)` | Calculates the coverage percentage (fraction of actuals within prediction interval). |
| `CalculateQuantileLoss(Tensor<>,Tensor<>)` | Calculates the quantile loss (pinball loss) for the predictions. |
| `CalculateQuantileLossGradient(Tensor<>,Tensor<>)` | Calculates the gradient of quantile loss for backpropagation. |
| `ComputeMultiQuantilePinballLossTape(Tensor<>,Tensor<>)` | Tape-aware multi-quantile pinball loss. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors for extended horizons. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads MQCNN-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the MQCNN model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `ExtractQuantilePrediction(Tensor<>,Double)` | Extracts predictions for a specific quantile level. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through MQCNN. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for MQCNN. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes MQCNN-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by incorporating predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet MQCNN architectural requirements. |
| `ValidateOptions(MQCNNOptions<>)` | Validates the MQCNN options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextLayer` | Context projection layer. |
| `_decoderLayers` | Decoder layers for quantile prediction. |
| `_encoderInputProjection` | Encoder input projection layer. |
| `_encoderLayers` | Encoder convolution layers for temporal pattern extraction. |
| `_outputLayer` | Final output layer producing quantile predictions. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |


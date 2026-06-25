---
title: "LagLlama<T>"
description: "Lag-Llama foundation model for probabilistic time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

Lag-Llama foundation model for probabilistic time series forecasting.

## For Beginners

Lag-Llama brings LLM innovations to time series forecasting:

**The Lag Feature Innovation:**
Instead of just looking at recent values, Lag-Llama creates features from specific past points:

- Lag-1: Yesterday's value (immediate trend)
- Lag-7: Same day last week (weekly pattern)
- Lag-365: Same day last year (annual pattern)

This lets the model explicitly see patterns at different time scales without needing
a very long context window.

**Llama Architecture Adaptations:**
Lag-Llama adopts key Llama innovations:

- **RMSNorm**: Simpler, faster layer normalization
- **SwiGLU**: Improved MLP activation function
- **RoPE**: Rotary Position Embeddings for better position encoding
- **Causal Attention**: Each position only sees earlier positions

**Probabilistic Forecasting:**
Unlike models that output single values, Lag-Llama outputs distribution parameters:

- For Student-t: degrees of freedom (nu), location (mu), scale (sigma)
- Allows uncertainty quantification: "The forecast is 100 ± 15"
- Enables risk-aware decisions: "There's a 5% chance it exceeds 130"

**Why Student-t Distribution?**

- Has heavier tails than Normal (better for extreme events)
- Degrades gracefully to Normal as nu → infinity
- More robust to outliers in training data

**Zero-Shot Capability:**
Pre-trained on diverse time series, Lag-Llama can forecast new series without training.

## How It Works

Lag-Llama adapts the Llama large language model architecture for time series forecasting.
It uses lag-based features to capture temporal patterns at multiple scales and outputs
probabilistic forecasts via distribution parameter prediction.

**Reference:** Rasul et al., "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting", 2024.
https://arxiv.org/abs/2310.08278

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LagLlama(NeuralNetworkArchitecture<>,LagLlamaOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Lag-Llama model in native mode for training or fine-tuning. |
| `LagLlama(NeuralNetworkArchitecture<>,String,LagLlamaOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Lag-Llama model using pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistributionOutput` | Gets the distribution type used for probabilistic output. |
| `IsChannelIndependent` |  |
| `LagIndices` | Gets the lag indices used for feature extraction. |
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
| `ApproximateInverseNormal(Double)` | Approximates the inverse standard normal CDF. |
| `AutoregressiveForecast(Tensor<>,Int32)` |  |
| `BuildLagFeatures(Tensor<>)` | Builds Lag-Llama's lag-feature representation (Rasul et al. |
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors for extended horizons. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads Lag-Llama-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the Lag-Llama model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `ExtractPointPredictions(Tensor<>)` | Extracts point predictions (means) from distribution parameters. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through Lag-Llama. |
| `ForwardNativeForTraining(Tensor<>)` | Training-mode forward pass. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for Lag-Llama. |
| `PredictCore(Tensor<>)` |  |
| `SampleQuantiles(Tensor<>,Double[])` | Samples quantiles from the predicted distribution. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes Lag-Llama-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by incorporating predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet Lag-Llama architectural requirements. |
| `ValidateOptions(LagLlamaOptions<>)` | Validates the Lag-Llama options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_distributionHead` | Distribution output head that predicts distribution parameters. |
| `_finalNorm` | Final layer normalization before output projection. |
| `_inputEmbedding` | Input embedding layer that projects lag features to hidden dimension. |
| `_transformerLayers` | Transformer layers with Llama-style architecture. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |


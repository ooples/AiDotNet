---
title: "DeepFactor<T>"
description: "DeepFactor (Deep Factor Model) for multivariate time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Neural`

DeepFactor (Deep Factor Model) for multivariate time series forecasting.

## For Beginners

DeepFactor is designed for forecasting many related time series:

**The Factor Model Idea:**
Many time series are driven by common underlying patterns:

- Stock prices: Market factors, sector factors, economic factors
- Retail sales: Holiday effects, weather, economic conditions
- Energy demand: Temperature, time of day, day of week

**The Decomposition:**
y_t = (factor_loadings * global_factors_t) + local_t + noise

- Global factors: Shared patterns learned from all series
- Factor loadings: How much each series is affected by each factor
- Local component: Series-specific patterns not captured by factors

**Why Deep Learning?**
Traditional factor models use linear relationships.
DeepFactor uses neural networks to:

- Learn non-linear factor dynamics (factors can evolve in complex ways)
- Automatically discover the right number and type of factors
- Capture complex interactions between factors

**Architecture:**

1. **Factor Model**: RNN that generates global factor values over time
2. **Loading Layer**: Maps factors to series-specific contributions
3. **Local Model**: Smaller network for series-specific patterns
4. **Combination**: Merges factor-based and local predictions

**Benefits:**

- Efficient: Shares computation across many series via factors
- Interpretable: Factors can be analyzed to understand shared patterns
- Robust: Less overfitting when series share common dynamics

## How It Works

DeepFactor combines classical factor models with deep learning. It decomposes time series
into global factors (shared patterns) and local components (series-specific behavior),
learning both through neural networks for improved forecasting accuracy.

**Reference:** Wang et al., "Deep Factors for Forecasting", 2019.
https://arxiv.org/abs/1905.12417

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepFactor(NeuralNetworkArchitecture<>,DeepFactorOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a DeepFactor model in native mode for training from scratch. |
| `DeepFactor(NeuralNetworkArchitecture<>,String,DeepFactorOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a DeepFactor model using pretrained ONNX model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsChannelIndependent` |  |
| `NumFactors` | Gets the number of latent factors in the model. |
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
| `ConcatenatePredictions(List<Tensor<>>,Int32)` | Concatenates multiple prediction tensors for extended horizons. |
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates two tensors along the last dimension. |
| `CreateNewInstance` | Creates a new instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads DeepFactor-specific configuration during deserialization. |
| `Dispose(Boolean)` | Disposes resources used by the DeepFactor model. |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `ExtractLayerReferences` | Extracts references to specific layers from the layer collection. |
| `Forecast(Tensor<>,Double[])` |  |
| `ForecastNative(Tensor<>)` | Performs native mode forecasting. |
| `ForecastOnnx(Tensor<>)` | Performs ONNX mode forecasting. |
| `Forward(Tensor<>)` | Performs the forward pass through DeepFactor. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for DeepFactor. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes DeepFactor-specific configuration during serialization. |
| `ShiftInputWithPredictions(Tensor<>,Tensor<>,Int32)` | Shifts input tensor by incorporating predictions for autoregressive forecasting. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that custom layers meet DeepFactor architectural requirements. |
| `ValidateOptions(DeepFactorOptions<>)` | Validates the DeepFactor options. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_combinationLayer` | Combination layer that merges factor and local predictions. |
| `_factorGenerationLayer` | Layer that generates factor values for the forecast horizon. |
| `_factorInputProjection` | Input projection layer for the factor model. |
| `_factorLoadingLayer` | Factor loading layer that maps factors to predictions. |
| `_factorRnnLayers` | RNN layers for the global factor model. |
| `_localInputProjection` | Input projection for the local model. |
| `_localLayers` | Layers for the local (series-specific) model. |
| `_localPredictionLayer` | Local prediction layer. |
| `_useNativeMode` | Indicates whether this network uses native layers (true) or ONNX model (false). |


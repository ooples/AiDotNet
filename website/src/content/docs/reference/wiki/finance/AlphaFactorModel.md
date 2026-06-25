---
title: "AlphaFactorModel<T>"
description: "Neural network model for learning alpha factors from market data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Factors`

Neural network model for learning alpha factors from market data.

## For Beginners

Think of this model as a factor "discoverer."
Instead of manually choosing factors like value or momentum, the model learns hidden
drivers of returns from historical data and then uses them to predict performance.

## How It Works

AlphaFactorModel learns latent factors that explain and predict excess returns.
It discovers these factors directly from data instead of relying on hand-crafted signals.

Reference: Chen et al. (2020). "Deep Learning for Alpha Generation"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AlphaFactorModel(NeuralNetworkArchitecture<>,AlphaFactorOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new AlphaFactorModel in native mode for training and inference. |
| `AlphaFactorModel(NeuralNetworkArchitecture<>,String,AlphaFactorOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new AlphaFactorModel in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumAssets` | Gets the number of assets covered by the model. |
| `NumFactors` | Gets the number of latent factors learned by the model. |
| `NumFeatures` | Gets the number of input features per asset. |
| `PredictionHorizon` | Gets the prediction horizon. |
| `SequenceLength` | Gets the input sequence length. |
| `SupportsTraining` | Gets whether training is supported in the current mode. |
| `UseNativeMode` | Gets whether the model is using native layers (true) or ONNX inference (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAlpha(Tensor<>,Tensor<>)` | Computes alpha (excess return) for each asset. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes model-specific data. |
| `Dispose(Boolean)` | Releases resources used by the model. |
| `ExtractFactors(Tensor<>)` | Extracts latent factors from asset returns. |
| `Forecast(Tensor<>,Double[])` | Generates a forecast using the model. |
| `GetFactorCovariance(Tensor<>)` | Computes the factor covariance matrix. |
| `GetFactorLoadings(Tensor<>)` | Computes factor loadings for each asset. |
| `GetFactorMetrics` | Gets factor model metrics. |
| `GetFinancialMetrics` | Gets financial metrics for the model. |
| `GetModelMetadata` | Gets metadata describing this model instance. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers for AlphaFactorModel. |
| `PredictCore(Tensor<>)` | Runs a forward pass to predict alpha values. |
| `PredictNative(Tensor<>)` | Runs a forward pass using native layers. |
| `PredictOnnx(Tensor<>)` | Runs a forward pass using the ONNX runtime. |
| `PredictReturns(Tensor<>)` | Predicts expected returns from factor exposures. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes model-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a batch of inputs and targets. |
| `UpdateParameters(Vector<>)` | Updates model parameters from a flat vector. |


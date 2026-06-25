---
title: "FactorVAE<T>"
description: "Variational autoencoder for learning disentangled financial factors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Factors`

Variational autoencoder for learning disentangled financial factors.

## For Beginners

The model compresses market data into a small set of hidden
variables (factors). The disentanglement penalty encourages each factor to capture
a different driver of returns rather than mixing everything together.

## How It Works

FactorVAE combines a variational autoencoder with a disentanglement penalty
so each latent dimension captures a distinct factor.

Reference: Kim & Mnih (2019). "Disentangling by Factorising"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FactorVAE(NeuralNetworkArchitecture<>,FactorVAEOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new FactorVAE in native mode for training and inference. |
| `FactorVAE(NeuralNetworkArchitecture<>,String,FactorVAEOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new FactorVAE in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LatentDimension` | Gets the dimension of the latent space. |
| `NumAssets` | Gets the number of assets covered by the model. |
| `NumFactors` | Gets the number of latent factors learned by the model. |
| `NumFeatures` | Gets the number of input features. |
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
| `InitializeLayers` | Initializes the neural network layers for FactorVAE. |
| `PredictCore(Tensor<>)` | Runs a forward pass to reconstruct inputs or generate factor outputs. |
| `PredictNative(Tensor<>)` | Runs a forward pass using native layers. |
| `PredictOnnx(Tensor<>)` | Runs a forward pass using the ONNX runtime. |
| `PredictReturns(Tensor<>)` | Predicts expected returns from factor exposures. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes model-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a batch of inputs and targets. |
| `UpdateParameters(Vector<>)` | Updates model parameters from a flat vector. |


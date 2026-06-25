---
title: "FactorTransformer<T>"
description: "Transformer-based model for learning financial factors with attention mechanisms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Factors`

Transformer-based model for learning financial factors with attention mechanisms.

## For Beginners

Transformers can focus on different parts of the data at once.
This model uses that ability to find hidden factor signals that drive asset returns.

## How It Works

FactorTransformer uses self-attention to model cross-sectional and temporal
relationships in financial data, extracting factors that capture complex patterns.

Reference: Duan et al. (2022). "FactorFormer: A Transformer-based Framework for Factor Investing"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FactorTransformer(NeuralNetworkArchitecture<>,FactorTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new FactorTransformer in native mode for training and inference. |
| `FactorTransformer(NeuralNetworkArchitecture<>,String,FactorTransformerOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new FactorTransformer in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumAssets` | Gets the number of assets covered by the model. |
| `NumFactors` | Gets the number of latent factors learned by the model. |
| `NumFeatures` | Gets the number of input features. |
| `NumHeads` | Gets the number of attention heads. |
| `NumTransformerLayers` | Gets the number of transformer layers. |
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
| `InitializeLayers` | Initializes the neural network layers for FactorTransformer. |
| `PredictCore(Tensor<>)` | Runs a forward pass to predict factor outputs. |
| `PredictNative(Tensor<>)` | Runs a forward pass using native layers. |
| `PredictOnnx(Tensor<>)` | Runs a forward pass using the ONNX runtime. |
| `PredictReturns(Tensor<>)` | Predicts expected returns from factor exposures. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes model-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a batch of inputs and targets. |
| `UpdateParameters(Vector<>)` | Updates model parameters from a flat vector. |


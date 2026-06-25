---
title: "TOTEM<T>"
description: "TOTEM — TOkenized Time Series EMbeddings via VQ-VAE."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Forecasting.Foundation`

TOTEM — TOkenized Time Series EMbeddings via VQ-VAE.

## For Beginners

TOTEM converts continuous time series data into discrete tokens
(like words in a vocabulary), making it possible to use language model techniques on
numerical data. Think of it as creating a "dictionary" of common time series patterns:
each chunk of data gets matched to its closest dictionary entry, creating a compact
representation that language-style models can process.

## How It Works

TOTEM learns discrete tokenized representations for time series via VQ-VAE,
enabling the use of discrete token-based methods (like LLMs) on continuous time series data.
It uses an encoder-decoder architecture with vector quantization bottleneck.

**Reference:** Talukder et al., "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TOTEM(NeuralNetworkArchitecture<>,String,TOTEMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TOTEM model using a pretrained ONNX model. |
| `TOTEM(NeuralNetworkArchitecture<>,TOTEMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a TOTEM model in native mode for training or fine-tuning. |

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
| `DenormalizeForecast(Tensor<>)` | RevIN reverse step (Kim et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Evaluate(Tensor<>,Tensor<>)` |  |
| `Forecast(Tensor<>,Double[])` |  |
| `ForwardNative(Tensor<>)` | VQ-VAE forward pass: encode → transformer → project to codebook dim → vector quantize (nearest neighbor lookup) → decode → forecast. |
| `ForwardNativeForTrainingWithCommitment(Tensor<>)` | Training-mode forward. |
| `GetCodebookValue(Int32,Int32,Int32)` | Gets a codebook value at the given indices. |
| `GetFinancialMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeCodebooks` | Initializes the VQ codebook embeddings with random values from N(0, 1/dim). |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetCodebookValue(Int32,Int32,Int32,)` | Sets a codebook value at the given indices. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `VectorQuantize(Tensor<>)` | Vector quantization: for each codebook, find nearest embedding to each input vector. |


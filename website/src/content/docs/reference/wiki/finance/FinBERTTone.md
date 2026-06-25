---
title: "FinBERTTone<T>"
description: "FinBERT-tone neural network model specialized for financial sentiment analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

FinBERT-tone neural network model specialized for financial sentiment analysis.

## For Beginners

FinBERT-Tone is a specialized sentiment analysis model for
financial text. Given a sentence from an earnings call, news article, or analyst report,
it classifies the sentiment as positive, negative, or neutral. It understands financial
context, so "revenue declined less than expected" is correctly identified as positive
sentiment despite containing the word "declined".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinBERTTone(NeuralNetworkArchitecture<>,FinBERTToneOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a FinBERT-tone network in native mode for training. |
| `FinBERTTone(NeuralNetworkArchitecture<>,String,FinBERTToneOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FinBERT-tone network using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the FinBERTTone. |
| `DeserializeModelSpecificData(BinaryReader)` | Executes DeserializeModelSpecificData for the FinBERTTone. |
| `ExtractLayerReferences` | Executes ExtractLayerReferences for the FinBERTTone. |
| `ForecastNative(Tensor<>,Double[])` | Executes ForecastNative for the FinBERTTone. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the FinBERTTone. |
| `SerializeModelSpecificData(BinaryWriter)` | Executes SerializeModelSpecificData for the FinBERTTone. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Executes TrainCore for the FinBERTTone. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the FinBERTTone. |
| `ValidateInputShape(Tensor<>)` | Executes ValidateInputShape for the FinBERTTone. |


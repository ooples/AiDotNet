---
title: "FinancialBERT<T>"
description: "FinancialBERT neural network model for domain-specific financial language processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

FinancialBERT neural network model for domain-specific financial language processing.

## For Beginners

FinancialBERT is a BERT model pre-trained on financial text
from corporate reports, analyst notes, and financial news. It understands financial
terminology and context better than general-purpose language models. Use it for tasks
like classifying financial documents, extracting key information from earnings reports,
or detecting sentiment in market commentary.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialBERT(NeuralNetworkArchitecture<>,FinancialBERTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a FinancialBERT network in native mode for training. |
| `FinancialBERT(NeuralNetworkArchitecture<>,String,FinancialBERTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FinancialBERT network using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the FinancialBERT. |
| `DeserializeModelSpecificData(BinaryReader)` | Executes DeserializeModelSpecificData for the FinancialBERT. |
| `ExtractLayerReferences` | Executes ExtractLayerReferences for the FinancialBERT. |
| `ForecastNative(Tensor<>,Double[])` | Executes ForecastNative for the FinancialBERT. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the FinancialBERT. |
| `SerializeModelSpecificData(BinaryWriter)` | Executes SerializeModelSpecificData for the FinancialBERT. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Executes TrainCore for the FinancialBERT. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the FinancialBERT. |
| `ValidateInputShape(Tensor<>)` | Executes ValidateInputShape for the FinancialBERT. |


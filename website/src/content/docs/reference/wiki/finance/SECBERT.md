---
title: "SECBERT<T>"
description: "SEC-BERT neural network model for domain-specific financial language processing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

SEC-BERT neural network model for domain-specific financial language processing.

## For Beginners

SEC-BERT is a language model trained exclusively on SEC
filings (10-K annual reports, 10-Q quarterly reports, 8-K current reports). It understands
the unique language and structure of regulatory documents, making it ideal for extracting
information from corporate disclosures, identifying risk factors, and classifying
financial statements.

## How It Works

SEC-BERT is a BERT-based model specifically pretrained on SEC filings (10-K, 10-Q, etc.).

Reference: Loukas et al., "SEC-BERT: A Pre-trained Financial Language Model", 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SECBERT(NeuralNetworkArchitecture<>,SECBERTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a SEC-BERT network in native mode for training. |
| `SECBERT(NeuralNetworkArchitecture<>,String,SECBERTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SEC-BERT network using a pretrained ONNX model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Executes CreateNewInstance for the SECBERT. |
| `DeserializeModelSpecificData(BinaryReader)` | Executes DeserializeModelSpecificData for the SECBERT. |
| `ExtractLayerReferences` | Executes ExtractLayerReferences for the SECBERT. |
| `ForecastNative(Tensor<>,Double[])` | Executes ForecastNative for the SECBERT. |
| `GetOptions` |  |
| `InitializeLayers` | Executes InitializeLayers for the SECBERT. |
| `SerializeModelSpecificData(BinaryWriter)` | Executes SerializeModelSpecificData for the SECBERT. |
| `TrainCore(Tensor<>,Tensor<>,Tensor<>)` | Executes TrainCore for the SECBERT. |
| `UpdateParameters(Vector<>)` | Executes UpdateParameters for the SECBERT. |
| `ValidateInputShape(Tensor<>)` | Executes ValidateInputShape for the SECBERT. |


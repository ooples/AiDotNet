---
title: "PICK<T>"
description: "PICK (Processing Key Information Extraction) neural network for document key information extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.GraphBased`

PICK (Processing Key Information Extraction) neural network for document key information extraction.

## For Beginners

PICK is especially good at:

1. Extracting key-value pairs from invoices and receipts
2. Understanding relationships between text segments
3. Handling complex document layouts
4. Named Entity Recognition in documents

Example usage:

## How It Works

PICK uses a graph neural network approach to extract key information from documents.
It models text segments as nodes and their relationships as edges, enabling
better understanding of document structure.

**Reference:** "PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks" (ICPR 2020)
https://arxiv.org/abs/2004.07464

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PICK(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,PICKOptions)` | Creates a PICK model using native layers for training and inference. |
| `PICK(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,PICKOptions)` | Creates a PICK model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedEntityTypes` | Gets the supported entity types for extraction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies PICK's industry-standard postprocessing: pass-through (entity extraction outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies PICK's industry-standard preprocessing: pass-through (PICK works with text + bbox input). |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectCheckboxes(Tensor<>)` |  |
| `DetectSignatures(Tensor<>)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractFormFields(Tensor<>)` |  |
| `ExtractFormFields(Tensor<>,Double)` |  |
| `ExtractKeyInfo(Tensor<>)` | Extracts key information entities from a document. |
| `ExtractKeyValuePairs(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |


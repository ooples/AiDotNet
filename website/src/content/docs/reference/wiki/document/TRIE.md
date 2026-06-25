---
title: "TRIE<T>"
description: "TRIE (Text Reading and Information Extraction) for end-to-end document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.GraphBased`

TRIE (Text Reading and Information Extraction) for end-to-end document understanding.

## For Beginners

TRIE does reading and extraction together:

1. Reads text from document images
2. Builds a graph of text entities
3. Extracts key-value pairs and entities
4. Outputs structured information

Key features:

- End-to-end text reading + extraction
- Graph-based entity relationship modeling
- Joint optimization of OCR and IE
- Strong performance on receipts and forms

Example usage:

## How It Works

TRIE combines text reading (OCR) with information extraction in an end-to-end framework,
using graph neural networks to model relationships between text entities and extract
structured information.

**Reference:** "TRIE: End-to-End Text Reading and Information Extraction" (ACM MM 2020)
https://arxiv.org/abs/2005.13118

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TRIE` | Creates a TRIE model with default configuration for native training. |
| `TRIE(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,TRIEOptions)` | Creates a TRIE model using native layers for training and inference. |
| `TRIE(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,TRIEOptions)` | Creates a TRIE model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `MinTextHeight` |  |
| `NumEntityTypes` | Gets the number of entity types. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportsPolygonOutput` |  |
| `SupportsRotatedText` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies TRIE's industry-standard postprocessing: pass-through (entity extraction outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies TRIE's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Decodes token IDs to text. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectCheckboxes(Tensor<>)` |  |
| `DetectSignatures(Tensor<>)` |  |
| `DetectText(Tensor<>)` |  |
| `DetectText(Tensor<>,Double)` |  |
| `DetectTextBatch(IEnumerable<Tensor<>>)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractBoundingBox(Tensor<>,Int32,Int32)` | Extracts bounding box coordinates from output. |
| `ExtractFieldText(Tensor<>,Int32,Int32,Int32,Int32)` | Extracts text from embedding portion of output. |
| `ExtractFormFields(Tensor<>)` |  |
| `ExtractFormFields(Tensor<>,Double)` |  |
| `ExtractKeyValuePairs(Tensor<>)` |  |
| `GetHeatmap` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `GetProbabilityMap(Tensor<>)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |


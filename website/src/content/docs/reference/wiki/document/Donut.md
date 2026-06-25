---
title: "Donut<T>"
description: "Donut (Document Understanding Transformer) - OCR-free end-to-end document understanding model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.PixelToSequence`

Donut (Document Understanding Transformer) - OCR-free end-to-end document understanding model.

## For Beginners

Unlike traditional document AI which first extracts text using OCR
and then processes it, Donut looks directly at the document image pixels and generates
text output. This makes it:

- Simpler: No need for a separate OCR system
- More robust: Less affected by OCR errors
- End-to-end trainable: Can optimize for the final task directly

Donut is excellent for:

- Document parsing (invoices, receipts, forms)
- Information extraction
- Document question answering
- Document classification

Example usage:

## How It Works

Donut is an OCR-free model that directly converts document images to structured text outputs
without requiring a separate OCR stage. It uses a vision encoder (Swin Transformer) and
text decoder (BART) architecture.

**Reference:** "OCR-free Document Understanding Transformer" (ECCV 2022)
https://arxiv.org/abs/2111.15664

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Donut(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32[],Int32[],Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DonutOptions)` | Creates a Donut model using native layers for training and inference. |
| `Donut(NeuralNetworkArchitecture<>,String,String,ITokenizer,Int32,Int32,Int32,Int32,Int32[],Int32[],Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DonutOptions)` | Creates a Donut model using pre-trained ONNX models for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `IsOCRFree` |  |
| `MaxGenerationLength` | Gets the maximum generation length for output sequences. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedLanguages` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies Donut's industry-standard postprocessing: pass-through (autoregressive outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies Donut's industry-standard preprocessing: normalize to [-1, 1]. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `ParseDocument(Tensor<>,String)` | Parses a document and returns structured output based on the document type. |
| `PredictCore(Tensor<>)` |  |
| `RebuildLayerGroupsFromLayers` | Re-derives the per-group mirror lists from the layers already present in `Layers` (e.g. |
| `RecognizeText(Tensor<>)` |  |
| `RecognizeTextInRegion(Tensor<>,Vector<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultImageHeight` | Creates a Donut model with default configuration for native training. |


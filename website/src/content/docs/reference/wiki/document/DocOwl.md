---
title: "DocOwl<T>"
description: "DocOwl (mPLUG-DocOwl) for document understanding with multimodal large language model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.VisionLanguage`

DocOwl (mPLUG-DocOwl) for document understanding with multimodal large language model.

## For Beginners

DocOwl brings LLM capabilities to documents:

1. Understands complex document layouts
2. Performs multi-page document understanding
3. Handles diverse document types (forms, tables, charts)
4. Generates detailed answers about document content

Key features:

- Based on mPLUG-Owl multimodal architecture
- Unified visual and text understanding
- Fine-tuned on document-specific datasets
- Strong generalization to unseen document types

Example usage:

## How It Works

DocOwl is based on the mPLUG-Owl architecture, specifically fine-tuned for document
understanding tasks. It combines a visual encoder with a large language model to
understand and reason about document content.

**Reference:** "mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding" (arXiv 2023)
https://arxiv.org/abs/2307.02499

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocOwl(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocOwlOptions)` | Creates a DocOwl model using native layers for training and inference. |
| `DocOwl(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocOwlOptions)` | Creates a DocOwl model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `LanguageDim` | Gets the language model dimension. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |
| `VisionDim` | Gets the vision encoder dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies DocOwl's industry-standard postprocessing: pass-through (multimodal LLM outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies DocOwl's industry-standard preprocessing: CLIP normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Converts token IDs to text using character-level decoding. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |


---
title: "InfographicVQA<T>"
description: "InfographicVQA for visual question answering on infographics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.VisionLanguage`

InfographicVQA for visual question answering on infographics.

## For Beginners

InfographicVQA specializes in complex visual documents:

1. Understands mixed content (text, charts, icons, diagrams)
2. Handles complex multi-column layouts
3. Performs visual reasoning across different element types
4. Extracts information from visually rich documents

Key features:

- Multi-scale visual processing
- OCR integration for text extraction
- Visual reasoning across elements
- Trained on InfographicsVQA dataset

Example usage:

## How It Works

InfographicVQA is designed to understand and answer questions about infographics,
which combine text, icons, charts, diagrams, and other visual elements in
complex layouts.

**Reference:** "InfographicVQA" (WACV 2022)
https://arxiv.org/abs/2104.12756

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InfographicVQA(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,InfographicVQAOptions)` | Creates an InfographicVQA model using native layers for training and inference. |
| `InfographicVQA(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,InfographicVQAOptions)` | Creates an InfographicVQA model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `FusionDim` | Gets the fusion dimension. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` | Gets the supported infographic element types. |
| `VisionDim` | Gets the vision encoder dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies InfographicVQA's industry-standard postprocessing: pass-through (VQA outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies InfographicVQA's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Converts token IDs to text using BERT-style vocabulary decoding. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
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


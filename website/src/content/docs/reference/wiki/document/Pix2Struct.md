---
title: "Pix2Struct<T>"
description: "Pix2Struct neural network for screenshot to structured output conversion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.PixelToSequence`

Pix2Struct neural network for screenshot to structured output conversion.

## For Beginners

Pix2Struct can:

1. Parse screenshots of web pages, charts, and documents
2. Extract structured information (tables, code, data)
3. Answer questions about visual content
4. Handle images of different sizes without resizing

Example usage:

## How It Works

Pix2Struct is a visually-situated language model that learns to parse screenshots
into structured outputs. It uses a variable-resolution vision encoder with ViT
and a text decoder to generate structured text from images.

**Reference:** "Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding" (ICML 2023)
https://arxiv.org/abs/2210.03347

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Pix2Struct(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Pix2StructOptions)` | Creates a Pix2Struct model using native layers for training and inference. |
| `Pix2Struct(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Pix2StructOptions)` | Creates a Pix2Struct model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies Pix2Struct's industry-standard postprocessing: pass-through (patch-to-text outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies Pix2Struct's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Converts token IDs to text using T5-style SentencePiece decoding. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `ParseScreenshot(Tensor<>,String)` | Parses a screenshot image and generates structured output. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |


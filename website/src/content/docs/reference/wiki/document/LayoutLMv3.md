---
title: "LayoutLMv3<T>"
description: "LayoutLMv3 neural network for document understanding with unified text and image pre-training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.LayoutAware`

LayoutLMv3 neural network for document understanding with unified text and image pre-training.

## For Beginners

LayoutLMv3 understands documents by learning from:

1. The text content (what the words say)
2. The visual appearance (what the document looks like)
3. The layout structure (where elements are positioned)

This makes it excellent for:

- Extracting information from forms and receipts
- Understanding document structure
- Answering questions about document content
- Classifying document types

Example usage (ONNX mode - for inference with pre-trained models):

Example usage (Native mode - for training):

## How It Works

LayoutLMv3 is the third generation of the LayoutLM series from Microsoft Research,
featuring unified multimodal pre-training with masked image modeling and masked language
modeling on the same architecture.

**Reference:** "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"
https://arxiv.org/abs/2204.08387

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayoutLMv3(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutLMv3Options)` | Creates a LayoutLMv3 model using native layers for training and inference. |
| `LayoutLMv3(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutLMv3Options)` | Creates a LayoutLMv3 model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies LayoutLMv3's industry-standard postprocessing: softmax for classification outputs. |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies LayoutLMv3's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `Forward(Tensor<>)` | Overrides Forward to handle LayoutLMv3's multimodal architecture. |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |


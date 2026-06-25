---
title: "LiLT<T>"
description: "LiLT (Language-Independent Layout Transformer) for document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.LayoutAware`

LiLT (Language-Independent Layout Transformer) for document understanding.

## For Beginners

LiLT is designed for maximum flexibility:

1. Layout understanding is learned separately from text
2. Can plug in ANY language model (BERT, RoBERTa, XLM-R, etc.)
3. Supports any language without retraining the layout part

Key features:

- BiACM (Bi-directional Attention Complementation Mechanism)
- Separate text and layout streams
- Works with any pre-trained text encoder
- Language-agnostic layout understanding

Example usage:

## How It Works

LiLT separates the text and layout modalities during pre-training, enabling
the layout model to be combined with ANY pre-trained text model at fine-tuning
time, providing true language independence.

**Reference:** "LiLT: A Simple yet Effective Language-Independent Layout Transformer" (ACL 2022)
https://arxiv.org/abs/2202.13669

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LiLT(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LiLTOptions)` | Creates a LiLT model using native layers for training and inference. |
| `LiLT(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LiLTOptions)` | Creates a LiLT model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |
| `TextBackbone` | Gets the text backbone model name. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies LiLT's industry-standard postprocessing: pass-through (layout-aware outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies LiLT's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractAnswer(Tensor<>,Int32,Double)` | Extracts answer from model output using a token-probability decoding pass. |
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


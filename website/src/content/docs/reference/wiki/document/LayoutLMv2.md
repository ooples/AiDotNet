---
title: "LayoutLMv2<T>"
description: "LayoutLMv2 neural network for document understanding with visual features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.LayoutAware`

LayoutLMv2 neural network for document understanding with visual features.

## For Beginners

LayoutLMv2 improves on v1 by also looking at the actual image:

1. Text content (what the words say)
2. Layout structure (where words are positioned)
3. Visual appearance (what the document looks like)

Key improvements over v1:

- Visual backbone (ResNeXt-FPN) for image features
- Spatial-aware self-attention mechanism
- Pre-training on both text-layout and image-text-layout alignment

Example usage:

## How It Works

LayoutLMv2 extends LayoutLM by adding visual features from a CNN backbone,
enabling the model to understand documents through text, layout, AND image features.

**Reference:** "LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding" (ACL 2021)
https://arxiv.org/abs/2012.14740

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayoutLMv2(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutLMv2Options)` | Creates a LayoutLMv2 model using native layers for training and inference. |
| `LayoutLMv2(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutLMv2Options)` | Creates a LayoutLMv2 model using a pre-trained ONNX model for inference. |

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
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies LayoutLMv2's industry-standard postprocessing: pass-through (multimodal outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies LayoutLMv2's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Decodes token IDs to text using BERT-style vocabulary. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractAnswer(Tensor<>,Int32)` | Extracts answer from model output using extractive QA approach. |
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


---
title: "LayoutXLM<T>"
description: "LayoutXLM neural network for multilingual document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.LayoutAware`

LayoutXLM neural network for multilingual document understanding.

## For Beginners

LayoutXLM understands documents in many languages:

1. Supports 53 languages out-of-the-box
2. Can handle mixed-language documents
3. Zero-shot cross-lingual transfer (train on one language, test on another)

Key features:

- XLM-RoBERTa multilingual text encoder
- Visual backbone (ResNeXt-FPN) for image features
- Language-agnostic layout understanding
- Pre-trained on XFUND dataset (7 languages)

Example usage:

## How It Works

LayoutXLM extends LayoutLMv2 to support multilingual documents by using XLM-RoBERTa
as the text backbone and training on documents from multiple languages.

**Reference:** "LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding" (ACL 2022)
https://arxiv.org/abs/2104.08836

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayoutXLM(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutXLMOptions)` | Creates a LayoutXLM model using native layers for training and inference. |
| `LayoutXLM(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutXLMOptions)` | Creates a LayoutXLM model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `NumLanguages` | Gets the number of languages supported. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies LayoutXLM's industry-standard postprocessing: pass-through (multilingual outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies LayoutXLM's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Decodes token IDs to text using BERT-style vocabulary. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractAnswer(Tensor<>,Int32)` | Extracts answer from model output using extractive QA approach. |
| `ExtractFields(Tensor<>,IEnumerable<String>)` |  |
| `ForwardForTraining(Tensor<>)` |  |
| `ForwardFromLayer(Tensor<>,Int32)` | Runs the layer chain starting at `startIndex` instead of layer zero — the text-only counterpart of `Tensor{` that lets the paper-supported text-stream-only operating mode bypass the visual backbone prefix. |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `VisualBackbonePrefixLength` | Number of layers in `Layers` that form the ResNeXt-FPN visual backbone (Conv7×7 → BN → MaxPool → Conv3×3 → visual-projection Dense). |


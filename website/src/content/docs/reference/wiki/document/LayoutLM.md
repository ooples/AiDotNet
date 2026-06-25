---
title: "LayoutLM<T>"
description: "LayoutLM (v1) neural network for document understanding with layout-aware pre-training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.LayoutAware`

LayoutLM (v1) neural network for document understanding with layout-aware pre-training.

## For Beginners

LayoutLM understands documents by learning from both:

1. The text content (what the words say)
2. The layout structure (where words are positioned on the page)

Unlike LayoutLMv2/v3, this version does NOT use visual features (images),
making it lighter but less powerful for visually-rich documents.

Example usage:

## How It Works

LayoutLM is the first generation of Microsoft's layout-aware document understanding models.
It combines text embeddings with 2D position embeddings to jointly model text and layout.

**Reference:** "LayoutLM: Pre-training of Text and Layout for Document Image Understanding" (KDD 2020)
https://arxiv.org/abs/1912.13318

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayoutLM(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutLMOptions)` | Creates a LayoutLM model using native layers for training and inference. |
| `LayoutLM(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,LayoutLMOptions)` | Creates a LayoutLM model using a pre-trained ONNX model for inference. |

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
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies LayoutLM's industry-standard postprocessing: softmax over class dimension. |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies LayoutLM's industry-standard preprocessing: pass-through (works with pre-tokenized input). |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectLayout(Tensor<>)` |  |
| `DetectLayout(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |


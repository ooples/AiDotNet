---
title: "Dessurt<T>"
description: "Dessurt (Document End-to-end Self-Supervised Understanding and RecogniTion) for document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.PixelToSequence`

Dessurt (Document End-to-end Self-Supervised Understanding and RecogniTion) for document understanding.

## For Beginners

Dessurt learns document understanding without labels:

1. Pre-trains by reconstructing corrupted document images
2. Learns to understand text, layout, and visual patterns
3. Fine-tunes on downstream tasks with minimal supervision

Key features:

- Self-supervised pre-training (no labels needed)
- Denoising autoencoder objective
- Vision encoder + text decoder architecture
- OCR-free document understanding

Example usage:

## How It Works

Dessurt is a self-supervised pre-training approach for document understanding that learns
from document images without any labeled data. It uses a denoising autoencoder objective
to learn robust document representations.

**Reference:** "Dessurt: A Dessert for Document Understanding" (arXiv 2022)
https://arxiv.org/abs/2203.16618

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Dessurt(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DessurtOptions)` | Creates a Dessurt model using native layers for training and inference. |
| `Dessurt(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DessurtOptions)` | Creates a Dessurt model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderDim` | Gets the decoder hidden dimension. |
| `EncoderDim` | Gets the encoder hidden dimension. |
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerQuestion(Tensor<>,String)` |  |
| `AnswerQuestion(Tensor<>,String,Int32,Double)` |  |
| `AnswerQuestions(Tensor<>,IEnumerable<String>)` |  |
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies Dessurt's industry-standard postprocessing: pass-through (sequence outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies Dessurt's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DecodeTokensToText(List<Int32>)` | Converts token IDs to text using character-level decoding. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `EnsureNativeInitialized` | Ensures native layers and embeddings are initialized on first use. |
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


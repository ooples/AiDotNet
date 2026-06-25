---
title: "DocFormer<T>"
description: "DocFormer neural network for end-to-end document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.LayoutAware`

DocFormer neural network for end-to-end document understanding.

## For Beginners

DocFormer combines three types of information:

1. Text content (what the words say)
2. Visual features (what the document looks like)
3. Spatial layout (where elements are positioned)

Unlike LayoutLM which adds position embeddings to text, DocFormer uses shared
spatial encodings that align all three modalities in the same coordinate space.

Example usage:

## How It Works

DocFormer is a multi-modal transformer that jointly learns text, visual, and spatial features
for document understanding tasks. It uses shared spatial encodings across all modalities.

**Reference:** "DocFormer: End-to-End Transformer for Document Understanding" (ICCV 2021)
https://arxiv.org/abs/2106.11539

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocFormer(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocFormerOptions)` | Creates a DocFormer model using native layers for training and inference. |
| `DocFormer(NeuralNetworkArchitecture<>,String,ITokenizer,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DocFormerOptions)` | Creates a DocFormer model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableCategories` | Gets the available document classification categories. |
| `ExpectedImageSize` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies DocFormer's industry-standard postprocessing: pass-through (multimodal outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies DocFormer's industry-standard preprocessing: ImageNet normalization. |
| `ClassifyDocument(Tensor<>)` |  |
| `ClassifyDocument(Tensor<>,Int32)` |  |
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


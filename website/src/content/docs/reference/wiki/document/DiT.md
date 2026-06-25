---
title: "DiT<T>"
description: "DiT (Document Image Transformer) for document image understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.LayoutAware`

DiT (Document Image Transformer) for document image understanding.

## For Beginners

DiT learns document understanding from images alone:

1. Pre-trains on 42 million document images
2. Uses masked image modeling (predicts missing patches)
3. Learns document-specific visual patterns

Key features:

- Pure vision approach (no OCR needed for pre-training)
- ViT-base/large architectures
- State-of-the-art on document classification
- Strong layout analysis performance

Example usage:

## How It Works

DiT applies self-supervised pre-training on large-scale document images using
a Vision Transformer (ViT) backbone, enabling strong document layout analysis
without requiring OCR annotations.

**Reference:** "DiT: Self-supervised Pre-training for Document Image Transformer" (ACM MM 2022)
https://arxiv.org/abs/2203.02378

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiT(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DiTOptions)` | Creates a DiT model using native layers for training and inference. |
| `DiT(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DiTOptions)` | Creates a DiT model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableCategories` |  |
| `ExpectedImageSize` |  |
| `ModelSize` | Gets the model size variant (base/large). |
| `PatchSize` | Gets the patch size for the ViT backbone. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportedElementTypes` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies DiT's industry-standard postprocessing: pass-through (classification outputs are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies DiT's industry-standard preprocessing: ImageNet normalization. |
| `ClassifyDocument(Tensor<>)` |  |
| `ClassifyDocument(Tensor<>,Int32)` |  |
| `ClassifyDocumentBatch(IEnumerable<Tensor<>>)` |  |
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


---
title: "PSENet<T>"
description: "PSENet (Progressive Scale Expansion Network) for text detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextDetection`

PSENet (Progressive Scale Expansion Network) for text detection.

## For Beginners

PSENet handles difficult text detection scenarios:

1. Detects text at multiple scales (kernels)
2. Progressively expands from smallest to largest
3. Separates closely spaced text instances
4. Handles arbitrary-shaped text

Key features:

- Multi-scale kernel prediction
- Progressive scale expansion algorithm
- Handles closely adjacent text
- Accurate boundary detection

Example usage:

## How It Works

PSENet uses a novel progressive scale expansion algorithm to accurately detect
text instances of various shapes and sizes, especially useful for closely spaced text.

**Reference:** "Shape Robust Text Detection with Progressive Scale Expansion Network" (CVPR 2019)
https://arxiv.org/abs/1903.12473

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PSENet` | Creates a PSENet model with default configuration for native training. |
| `PSENet(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,PSENetOptions)` | Creates a PSENet model using native layers for training and inference. |
| `PSENet(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,PSENetOptions)` | Creates a PSENet model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `MinTextHeight` |  |
| `NumKernels` | Gets the number of scale kernels. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportsPolygonOutput` |  |
| `SupportsRotatedText` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies PSENet's industry-standard postprocessing: pass-through (kernel maps are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies PSENet's industry-standard preprocessing: ImageNet normalization with scale. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectText(Tensor<>)` |  |
| `DetectText(Tensor<>,Double)` |  |
| `DetectTextBatch(IEnumerable<Tensor<>>)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `ExtractConvexHull(List<ValueTuple<Int32,Int32>>)` | Extracts convex hull from pixel coordinates (simplified algorithm). |
| `GetHeatmap` |  |
| `GetModelMetadata` |  |
| `GetModelSummary` |  |
| `GetOptions` |  |
| `GetProbabilityMap(Tensor<>)` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `ValidateInputShape(Tensor<>)` |  |


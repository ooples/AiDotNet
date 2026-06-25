---
title: "DBNet<T>"
description: "DBNet (Differentiable Binarization Network) for real-time text detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextDetection`

DBNet (Differentiable Binarization Network) for real-time text detection.

## For Beginners

DBNet finds where text is located in an image. It works by:

1. Creating a "probability map" showing how likely each pixel is to be text
2. Creating a "threshold map" that adapts to different text styles
3. Combining them into a "binary map" showing exact text regions

The key innovation is that the threshold is learned, not fixed, which helps with
various fonts, sizes, and backgrounds.

Example usage:

## How It Works

DBNet is a fast and accurate text detection model that uses differentiable binarization
to produce sharp text boundary predictions. It outputs probability, threshold, and binary maps.

**Reference:** "Real-time Scene Text Detection with Differentiable Binarization" (AAAI 2020)
https://arxiv.org/abs/1911.08947

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DBNet(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Double,Double,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DBNetOptions)` | Creates a DBNet model using native layers for training and inference. |
| `DBNet(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Double,Double,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DBNetOptions)` | Creates a DBNet model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `MinTextHeight` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportsPolygonOutput` |  |
| `SupportsRotatedText` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies DBNet's industry-standard postprocessing: pass-through (probability map is already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies DBNet's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectText(Tensor<>)` |  |
| `DetectText(Tensor<>,Double)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
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


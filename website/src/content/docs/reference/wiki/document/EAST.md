---
title: "EAST<T>"
description: "EAST (Efficient and Accurate Scene Text Detector) for text detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextDetection`

EAST (Efficient and Accurate Scene Text Detector) for text detection.

## For Beginners

EAST is designed for speed and accuracy:

1. Single-shot detection (no multi-stage pipeline)
2. Outputs rotated boxes or quadrilaterals
3. Very fast inference
4. Works on arbitrary text orientations

Key features:

- Fully convolutional architecture
- Multi-scale feature fusion
- Direct geometry prediction
- Efficient NMS

Example usage:

## How It Works

EAST is a fast and accurate scene text detector that directly predicts text regions
without requiring complex post-processing like NMS across multiple stages.

**Reference:** "EAST: An Efficient and Accurate Scene Text Detector" (CVPR 2017)
https://arxiv.org/abs/1704.03155

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EAST(NeuralNetworkArchitecture<>,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,EASTOptions)` | Creates an EAST model using native layers for training and inference. |
| `EAST(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,String,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,EASTOptions)` | Creates an EAST model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `GeometryType` | Gets the geometry output type (RBOX or QUAD). |
| `MinTextHeight` |  |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportsPolygonOutput` |  |
| `SupportsRotatedText` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies EAST's industry-standard postprocessing: pass-through (geometry maps are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies EAST's industry-standard preprocessing: VGG mean subtraction. |
| `ApplyNMS(List<TextRegion<>>,Double)` | Applies non-maximum suppression to remove overlapping detections. |
| `CalculateIoU(TextRegion<>,TextRegion<>)` | Calculates intersection over union between two regions. |
| `CalculateRotatedBox(Double,Double,Double,Double,Double)` | Calculates rotated bounding box corners. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectText(Tensor<>)` |  |
| `DetectText(Tensor<>,Double)` |  |
| `DetectTextBatch(IEnumerable<Tensor<>>)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `Forward(Tensor<>)` | Overrides Forward to handle EAST's parallel output heads (score map + geometry). |
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


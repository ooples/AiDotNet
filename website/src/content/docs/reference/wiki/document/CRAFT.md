---
title: "CRAFT<T>"
description: "CRAFT (Character Region Awareness for Text Detection) neural network for character-level text detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document.OCR.TextDetection`

CRAFT (Character Region Awareness for Text Detection) neural network for character-level text detection.

## For Beginners

CRAFT works by:

1. Finding each character individually (character region)
2. Finding the connections between characters (affinity)
3. Grouping connected characters into words/lines

This approach handles:

- Curved text
- Rotated text
- Dense text regions
- Multiple languages

Example usage:

## How It Works

CRAFT detects text at the character level by predicting character regions and affinity
(the relationship between characters) maps. This enables precise detection of text
with arbitrary shapes, orientations, and sizes.

**Reference:** "Character Region Awareness for Text Detection" (CVPR 2019)
https://arxiv.org/abs/1904.01941

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CRAFT(NeuralNetworkArchitecture<>,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,CRAFTOptions)` | Creates a CRAFT model using native layers for training and inference. |
| `CRAFT(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,IOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,CRAFTOptions)` | Creates a CRAFT model using a pre-trained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedImageSize` |  |
| `MinTextHeight` |  |
| `MinTextSize` | Gets the minimum supported text size. |
| `RequiresOCR` |  |
| `SupportedDocumentTypes` |  |
| `SupportsCharacterDetection` | Gets whether this detector supports character-level detection. |
| `SupportsLineDetection` | Gets whether this detector supports line-level detection. |
| `SupportsPolygonOutput` | Gets whether polygon output is supported. |
| `SupportsRotatedText` |  |
| `SupportsWordDetection` | Gets whether this detector supports word-level detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefaultPostprocessing(Tensor<>)` | Applies CRAFT's industry-standard postprocessing: pass-through (character and affinity maps are already final). |
| `ApplyDefaultPreprocessing(Tensor<>)` | Applies CRAFT's industry-standard preprocessing: ImageNet normalization. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectText(Tensor<>)` |  |
| `DetectText(Tensor<>,Double)` |  |
| `DetectTextBatch(IEnumerable<Tensor<>>)` |  |
| `Dispose(Boolean)` |  |
| `EncodeDocument(Tensor<>)` |  |
| `GetAffinityMap` | Gets the affinity (character connection) heatmap from the last detection. |
| `GetCharacterMap` | Gets the character region heatmap from the last detection. |
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


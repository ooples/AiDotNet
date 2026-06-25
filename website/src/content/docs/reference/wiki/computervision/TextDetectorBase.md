---
title: "TextDetectorBase<T>"
description: "Base class for text detection models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Detection.TextDetection`

Base class for text detection models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextDetectorBase(TextDetectionOptions<>)` | Creates a new text detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `EnsureBackbone` | Gets the backbone network, throwing if not initialized. |
| `Name` | Name of this text detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `Detect(Tensor<>)` | Detects text regions in an image. |
| `Detect(Tensor<>,Double)` | Detects text regions with custom threshold. |
| `FitMinAreaRect(List<ValueTuple<Double,Double>>)` | Converts polygon points to a minimum area rotated rectangle. |
| `Forward(Tensor<>)` | Forward pass through the network. |
| `GetHeadParameterCount` | Gets the parameter count of the detection head. |
| `GetParameterCount` | Gets the total parameter count of the model. |
| `GetParameters` |  |
| `LoadWeightsAsync(String,CancellationToken)` | Loads pretrained weights. |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double)` | Post-processes network outputs to get text regions. |
| `Predict(Tensor<>)` | Predicts by returning the preprocessed input (text detection is done via Detect method). |
| `Preprocess(Tensor<>)` | Preprocesses the input image. |
| `SaveWeights(String)` | Saves model weights. |
| `SetParameters(Vector<>)` |  |
| `SimplifyPolygon(List<ValueTuple<Double,Double>>,Double)` | Simplifies a polygon using Douglas-Peucker algorithm. |
| `Train(Tensor<>,Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |


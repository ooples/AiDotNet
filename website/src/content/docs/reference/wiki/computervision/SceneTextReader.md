---
title: "SceneTextReader<T>"
description: "End-to-end scene text reader that combines detection and recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.OCR.EndToEnd`

End-to-end scene text reader that combines detection and recognition.

## For Beginners

SceneTextReader is a complete OCR pipeline that first
detects text regions in images (using CRAFT, EAST, or DBNet), then recognizes
the text in each region (using CRNN or TrOCR). It's designed for reading text
in natural images like photos of signs, billboards, and product labels.

## How It Works

Key features:

- Two-stage pipeline: detection + recognition
- Handles arbitrary text orientations
- Works with curved and rotated text
- Configurable detection and recognition models

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SceneTextReader(OCROptions<>)` | Creates a new scene text reader with default models. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Name` | Name of this scene text reader. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BilinearSample(Tensor<>,Int32,Int32,Double,Double,Int32,Int32)` | Bilinear-interpolated sample from a [batch, channel, H, W] tensor at a fractional position. |
| `CorrectPerspective(Tensor<>,TextRegion<>,Int32,Int32)` | Removes perspective distortion from a text crop by warping the four polygon corners onto a canonical axis-aligned rectangle via a planar homography (DLT, Hartley & Zisserman 2003 §4.1). |
| `DeepCopy` |  |
| `GaussianEliminate(Double[0:,0:],Double[],Double[])` | Gaussian elimination with partial pivoting on an 8×8 system. |
| `GetParameterCount` | Gets the total parameter count. |
| `GetParameters` |  |
| `LoadWeightsAsync(String,String,CancellationToken)` | Loads pretrained weights. |
| `OrderQuadCorners(List<ValueTuple<,>>)` | Orders 4 polygon corners as (top-left, top-right, bottom-right, bottom-left) so the homography target rectangle has a consistent orientation. |
| `Predict(Tensor<>)` | Runs end-to-end OCR and returns region info as a tensor [numRegions, 6]. |
| `ReadRegion(Tensor<>)` | Reads text from a pre-detected region. |
| `ReadText(Tensor<>)` | Reads all text in an image. |
| `ReducePolygonToQuad(List<ValueTuple<,>>)` | Reduces a polygon with N >= 4 vertices to a stable 4-corner quad using the canonical extreme-point selection from OpenCV's perspective-correct pipeline: top-left = argmin(x + y) bottom-right = argmax(x + y) top-right = argmax(x - y) bottom-… |
| `SetParameters(Vector<>)` |  |
| `SolveHomographyDLT(Double[],Double[],Double[],Double[],Double[])` | Direct Linear Transform: solves for the 8 parameters of a planar homography mapping (dx_i, dy_i) → (sx_i, sy_i) for i ∈ {0..3}. |
| `Train(Tensor<>,Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |


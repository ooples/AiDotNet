---
title: "DBNet<T>"
description: "DBNet (Differentiable Binarization Network) text detector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.TextDetection`

DBNet (Differentiable Binarization Network) text detector.

## For Beginners

DBNet is a state-of-the-art text detector that uses
differentiable binarization to segment text regions. Unlike traditional methods
that use a fixed threshold, DBNet learns an adaptive threshold for each pixel,
making it more robust to varying text appearances.

## How It Works

Key features:

- Differentiable binarization for end-to-end training
- Adaptive thresholding per pixel
- Fast inference with single-pass architecture
- Works well for both regular and irregular text shapes

Reference: Liao et al., "Real-time Scene Text Detection with Differentiable
Binarization", AAAI 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DBNet(TextDetectionOptions<>,Double)` | Creates a new DBNet text detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` |  |
| `Detect(Tensor<>,Double)` |  |
| `Forward(Tensor<>)` |  |
| `GetHeadParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double)` |  |
| `SaveWeights(String)` |  |


---
title: "CRAFT<T>"
description: "CRAFT (Character Region Awareness for Text) detector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.TextDetection`

CRAFT (Character Region Awareness for Text) detector.

## For Beginners

CRAFT detects text by identifying individual characters
and the connections (affinity) between them. This approach works well for scene text
with arbitrary orientations and curved text.

## How It Works

Key features:

- Character region score map
- Affinity score map (character connections)
- Works with arbitrary shaped text
- Good for scene text detection

Reference: Baek et al., "Character Region Awareness for Text Detection", CVPR 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CRAFT(TextDetectionOptions<>)` | Creates a new CRAFT text detector. |

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


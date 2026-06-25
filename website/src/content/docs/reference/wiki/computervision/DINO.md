---
title: "DINO<T>"
description: "DINO (DETR with Improved deNoising anchOr boxes) - State-of-the-art DETR variant."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.ObjectDetection.DETR`

DINO (DETR with Improved deNoising anchOr boxes) - State-of-the-art DETR variant.

## For Beginners

DINO improves upon DETR by using contrastive denoising training
and mixed query selection. It achieves better performance with faster convergence than
the original DETR.

## How It Works

Key improvements:

- Contrastive denoising training for better query learning
- Mixed query selection (both content and position queries)
- Look forward twice for better box predictions
- Multi-scale deformable attention (optional)

Reference: Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection", ICLR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DINO(ObjectDetectionOptions<>)` | Creates a new DINO detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>,Double,Double)` |  |
| `Forward(Tensor<>)` |  |
| `GetHeadParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double,Double)` |  |
| `SaveWeights(String)` |  |


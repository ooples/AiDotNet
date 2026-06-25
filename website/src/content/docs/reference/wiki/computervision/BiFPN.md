---
title: "BiFPN<T>"
description: "Bidirectional Feature Pyramid Network (BiFPN) with weighted feature fusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Necks`

Bidirectional Feature Pyramid Network (BiFPN) with weighted feature fusion.

## For Beginners

BiFPN improves upon FPN and PANet by using learnable
weights for feature fusion. Instead of simply adding features, it learns how
much each input feature should contribute to the fused output.

## How It Works

Key features:

- Bidirectional (top-down + bottom-up) feature flow
- Learnable weights for weighted feature fusion
- Fast normalized fusion with softmax
- Used in EfficientDet for state-of-the-art detection

Reference: Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiFPN(Int32[],Int32,Int32)` | Creates a new Bidirectional Feature Pyramid Network. |
| `BiFPN(NeckConfig)` | Creates BiFPN from a configuration object. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `NumLevels` |  |
| `OutputChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CopyTensorInto(Tensor<>,Tensor<>)` | Copies every element from `src` into `dst` in native `T` arithmetic. |
| `DeepCopy` |  |
| `FastNormalizedFusion(List<Tensor<>>,List<Tensor<>>)` | Fast normalized fusion with learned weights. |
| `Forward(List<Tensor<>>)` |  |
| `GetParameterCount` |  |
| `ReadParameters(BinaryReader)` |  |
| `WriteParameters(BinaryWriter)` |  |


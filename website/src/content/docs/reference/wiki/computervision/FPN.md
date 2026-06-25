---
title: "FPN<T>"
description: "Feature Pyramid Network (FPN) for multi-scale feature fusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Necks`

Feature Pyramid Network (FPN) for multi-scale feature fusion.

## For Beginners

FPN creates a feature pyramid by combining features from
different backbone levels using a top-down pathway. High-resolution features from
earlier layers are enriched with semantic information from deeper layers.

## How It Works

Key features:

- Top-down pathway with lateral connections
- 1x1 convolutions to match channel dimensions
- Simple element-wise addition for fusion
- Fast and memory efficient

Reference: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FPN(Int32[],Int32)` | Creates a new Feature Pyramid Network. |
| `FPN(NeckConfig)` | Creates FPN from a configuration object. |

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
| `Forward(List<Tensor<>>)` |  |
| `GetParameterCount` |  |
| `ReadParameters(BinaryReader)` |  |
| `WriteParameters(BinaryWriter)` |  |


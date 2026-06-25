---
title: "PANet<T>"
description: "Path Aggregation Network (PANet) for enhanced multi-scale feature fusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Necks`

Path Aggregation Network (PANet) for enhanced multi-scale feature fusion.

## For Beginners

PANet improves upon FPN by adding a bottom-up pathway
after the top-down pathway. This creates a bidirectional flow of information,
allowing both high-level semantics to flow down and low-level details to flow up.

## How It Works

Key features:

- FPN-style top-down pathway
- Additional bottom-up pathway for better localization
- Used in YOLOv4, YOLOv5, and many modern detectors

Reference: Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PANet(Int32[],Int32)` | Creates a new Path Aggregation Network. |
| `PANet(NeckConfig)` | Creates PANet from a configuration object. |

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

## Fields

| Field | Summary |
|:-----|:--------|
| `_random` | Random number generator for weight initialization (created once, reused for all weights). |


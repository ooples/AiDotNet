---
title: "SwinTransformer<T>"
description: "Swin Transformer backbone for hierarchical vision transformer feature extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Backbones`

Swin Transformer backbone for hierarchical vision transformer feature extraction.

## How It Works

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwinTransformer(SwinVariant,Int32,Int32)` | Creates a new Swin Transformer backbone. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Sum across patch embedding + every Swin stage. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `EnsureBatchedNchw(Tensor<>)` | Normalizes an image tensor to batched NCHW. |

## Fields

| Field | Summary |
|:-----|:--------|
| `PatchEmbeddingStride` | Patch-embedding stride. |


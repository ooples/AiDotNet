---
title: "MultiViewUNet<T>"
description: "Multi-view aware U-Net for MVDream."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Multi-view aware U-Net for MVDream.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiViewUNet(Int32,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Creates a new multi-view U-Net. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseUNet` | Gets the base UNet for interface compatibility. |
| `ParameterCount` | Gets parameter count. |
| `SupportsCFG` | Gets whether classifier-free guidance is supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameters` | Gets parameters. |
| `PredictNoise(Tensor<>,Int32,Tensor<>)` | Predicts noise for a single view. |
| `PredictNoiseMultiView(Tensor<>,Int32,Tensor<>[])` | Predicts noise for multiple views with cross-view attention. |
| `SetParameters(Vector<>)` | Sets parameters. |


---
title: "ImageProjector<T>"
description: "Projects image features to text embedding space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

Projects image features to text embedding space.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageProjector(Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new ImageProjector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Number of trainable parameters across the projection + token-expansion layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameters` | Gets all parameters. |
| `Project(Tensor<>)` | Projects image features to IP embedding. |
| `SetParameters(Vector<>)` | Sets all parameters. |


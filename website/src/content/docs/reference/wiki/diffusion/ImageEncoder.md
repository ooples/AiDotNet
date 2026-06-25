---
title: "ImageEncoder<T>"
description: "Image encoder for extracting features from reference images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

Image encoder for extracting features from reference images.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageEncoder(Int32,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new ImageEncoder. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Number of trainable parameters across patch projection + transformer layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Tensor<>)` | Encodes an image into feature embeddings. |
| `FlattenPatches(Tensor<>)` | Splits a `[1, 3, imageSize, imageSize]` image into `[numPatches, patchDim]` row-major patch tokens. |
| `GetParameters` | Gets all parameters. |
| `SetParameters(Vector<>)` | Sets all parameters. |


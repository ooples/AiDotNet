---
title: "StructuralSimilarity<T>"
description: "Structural Similarity Index Measure (SSIM) for image quality assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Structural Similarity Index Measure (SSIM) for image quality assessment.

## How It Works

SSIM measures structural similarity between two images, considering luminance, contrast, and structure.
SSIM values range from -1 to 1, where 1 indicates perfect similarity.

Formula: SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
where l = luminance, c = contrast, s = structure comparisons.

**Usage in 3D AI:**

- NeRF novel view synthesis evaluation
- Better perceptual quality metric than PSNR
- Captures structural distortions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StructuralSimilarity(Nullable<>,Double,Double,Int32)` | Initializes a new instance of the SSIM metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes SSIM between predicted and ground truth images. |
| `ComputeMean(Tensor<>)` | Computes the mean of a tensor. |
| `ComputeSingleChannel(Tensor<>,Tensor<>)` | Computes SSIM for a single channel image using sliding window. |
| `ExtractChannel(Tensor<>,Int32)` | Extracts a single channel from a multi-channel image. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_c1` | Stabilization constant for luminance (C1 = (K1 * L)^2). |
| `_c2` | Stabilization constant for contrast (C2 = (K2 * L)^2). |
| `_numOps` | The numeric operations provider for type T. |
| `_windowSize` | Window size for local statistics computation. |


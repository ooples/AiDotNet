---
title: "PeakSignalToNoiseRatio<T>"
description: "Peak Signal-to-Noise Ratio (PSNR) metric for image quality assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Peak Signal-to-Noise Ratio (PSNR) metric for image quality assessment.

## How It Works

PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
Higher PSNR values indicate better image quality. Common ranges:

- >40 dB: Excellent quality (nearly indistinguishable from original)
- 30-40 dB: Good quality
- 20-30 dB: Acceptable quality
- <20 dB: Poor quality

Formula: PSNR = 10 * log10(MAX² / MSE) where MAX is the maximum possible pixel value.

**Usage in 3D AI:**

- NeRF novel view synthesis evaluation
- Gaussian Splatting rendering quality
- Image reconstruction quality assessment

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PeakSignalToNoiseRatio(Nullable<>)` | Initializes a new instance of the PSNR metric. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes PSNR between predicted and ground truth images. |
| `ComputeBatch(Tensor<>,Tensor<>)` | Computes PSNR for a batch of images. |
| `ComputeMSE(Tensor<>,Tensor<>)` | Computes Mean Squared Error between two tensors. |
| `ShapesMatch(Int32[],Int32[])` | Checks if two shapes match. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_maxValue` | Maximum possible pixel value (e.g., 1.0 for normalized images, 255 for 8-bit). |
| `_numOps` | The numeric operations provider for type T. |


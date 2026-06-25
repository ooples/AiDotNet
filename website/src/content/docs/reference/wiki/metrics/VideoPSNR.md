---
title: "VideoPSNR<T>"
description: "Video Peak Signal-to-Noise Ratio (VPSNR) - Frame-averaged PSNR for video quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Video Peak Signal-to-Noise Ratio (VPSNR) - Frame-averaged PSNR for video quality.

## How It Works

VPSNR computes PSNR for each frame of a video and returns statistics (mean, min, max).
It's a straightforward extension of image PSNR to video sequences.

Typical values:

- >40 dB: Excellent quality
- 30-40 dB: Good quality
- 20-30 dB: Acceptable quality
- <20 dB: Poor quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoPSNR(Nullable<>)` | Initializes a new instance of VideoPSNR. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>,Boolean)` | Computes VPSNR between predicted and ground truth videos. |
| `ComputeWithStats(Tensor<>,Tensor<>,Boolean)` | Computes VPSNR with detailed per-frame statistics. |


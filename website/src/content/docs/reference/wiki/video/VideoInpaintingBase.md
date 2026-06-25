---
title: "VideoInpaintingBase<T>"
description: "Base class for video inpainting models that fill in missing or masked regions in video sequences."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Video`

Base class for video inpainting models that fill in missing or masked regions in video sequences.

## For Beginners

Video inpainting is like a smart "eraser" for video. You can mark
areas you want to remove (like a watermark, a person, or damage), and the model fills
those areas with realistic content that matches the surrounding video. It uses information
from other frames to figure out what should be there, making the result look natural and
consistent across the whole video.

## How It Works

Video inpainting fills in missing, damaged, or unwanted regions in video while maintaining
temporal consistency. This base class provides:

- Binary mask handling for specifying regions to inpaint
- Temporal propagation utilities for consistent fills across frames
- Completion quality metrics (PSNR, SSIM within masked regions)

Derived classes implement specific architectures like STTN, FuseFormer, ProPainter, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoInpaintingBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the VideoInpaintingBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxMaskRatio` | Gets the maximum supported mask ratio (fraction of frame that can be masked). |
| `SupportsTemporalPropagation` | Gets whether this model supports temporal propagation for inpainting. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMaskedPSNR(Tensor<>,Tensor<>,Tensor<>)` | Computes PSNR only within masked regions for inpainting quality assessment. |
| `Inpaint(Tensor<>,Tensor<>)` | Inpaints masked regions in a video sequence. |
| `PredictCore(Tensor<>)` |  |
| `PropagateTemporally(Tensor<>,Tensor<>,List<Tensor<>>)` | Propagates known pixel values from neighboring frames to fill masked regions. |


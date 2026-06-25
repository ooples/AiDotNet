---
title: "MiDaS<T>"
description: "MiDaS: Towards Robust Monocular Depth Estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Depth`

MiDaS: Towards Robust Monocular Depth Estimation.

## For Beginners

MiDaS estimates depth from a single image (monocular depth estimation).
Unlike stereo vision which uses two cameras, MiDaS uses deep learning to predict depth
from visual cues like texture, perspective, and object sizes.

Key capabilities:

- Single image to depth map conversion
- Works on arbitrary images without camera parameters
- Outputs relative depth (closer objects have higher values)
- Robust across different scenes and domains

Example usage:

## How It Works

**Technical Details:**

- ViT-based encoder for feature extraction
- Multi-scale fusion decoder
- Scale and shift invariant loss for training
- Trained on diverse mixed datasets for robustness

**Reference:** "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
https://arxiv.org/abs/1907.01341

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MiDaS` | Creates a MiDaS model with default configuration. |
| `MiDaS(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,MiDaSVariant,MiDaSOptions)` | Creates a MiDaS model using native layers for training and inference. |
| `MiDaS(NeuralNetworkArchitecture<>,String,MiDaSVariant,MiDaSOptions)` | Creates a MiDaS model using a pretrained ONNX model for inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateDepth(Tensor<>)` | Estimates depth from an input image. |
| `EstimateDepthForVideo(List<Tensor<>>)` | Estimates depth for multiple video frames. |
| `GetOptions` |  |
| `NormalizeDepthMap(Tensor<>)` | Normalizes depth map to 0-1 range for visualization. |


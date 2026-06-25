---
title: "RealESRGANVideoOptions"
description: "Configuration options for Real-ESRGAN extended to video with temporal consistency."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for Real-ESRGAN extended to video with temporal consistency.

## For Beginners

Real-ESRGAN is one of the most popular practical upscaling tools.
The video version adds temporal awareness so that when you upscale a video, each frame
looks consistent with its neighbors (no flickering). It uses a realistic degradation
model during training, so it handles real-world issues like compression artifacts,
noise, and blur that lab models struggle with.

## How It Works

Real-ESRGAN Video (Wang et al., 2022) extends the image-based Real-ESRGAN to video:

- RRDB backbone: Residual-in-Residual Dense Blocks (RRDBs) from ESRGAN provide the

per-frame feature extraction with strong representational capacity

- Second-order degradation model: simulates realistic degradations by applying

blur-resize-noise-JPEG twice in sequence, covering a wider range of real-world artifacts

- Temporal consistency module: flow-guided feature alignment between adjacent frames

with a temporal aggregation layer that fuses aligned features

- U-Net discriminator: a U-Net-based discriminator provides both global and local

adversarial feedback for high-quality perceptual results

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealESRGANVideoOptions` | Initializes a new instance with default values. |
| `RealESRGANVideoOptions(RealESRGANVideoOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DenseLayersPerBlock` | Gets or sets the number of dense layers per RRDB. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `GANWeight` | Gets or sets the weight for GAN adversarial loss. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumRRDBBlocks` | Gets or sets the number of RRDB (Residual-in-Residual Dense Block) blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PerceptualWeight` | Gets or sets the weight for perceptual (LPIPS) loss. |
| `ResidualScale` | Gets or sets the residual scaling factor for RRDB stability. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |


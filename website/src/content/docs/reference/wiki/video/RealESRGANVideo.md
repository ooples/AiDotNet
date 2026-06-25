---
title: "RealESRGANVideo<T>"
description: "Real-ESRGAN Video: practical real-world video super-resolution with temporal consistency."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

Real-ESRGAN Video: practical real-world video super-resolution with temporal consistency.

## For Beginners

Real-ESRGAN is one of the most widely-used practical upscaling
tools. The video version adds temporal awareness so each upscaled frame looks consistent
with its neighbors (no flickering). The key innovation is training with a "double
degradation" model -- it learns to handle all the messy artifacts (compression, noise,
blur) that real videos have, not just the simple downscaling used in lab benchmarks.

**Usage:**

## How It Works

Real-ESRGAN Video (Wang et al., 2022) extends the image-based Real-ESRGAN to video:

- RRDB backbone: Residual-in-Residual Dense Blocks provide per-frame feature extraction

with strong representational capacity from densely connected layers

- Second-order degradation model: training simulates realistic degradations by applying

blur-resize-noise-JPEG twice in sequence, covering a much wider range of real-world
artifacts than first-order models

- Temporal consistency module: flow-guided feature alignment between adjacent frames

followed by temporal aggregation that fuses aligned features with learned attention

- U-Net discriminator: provides both global structure feedback and local detail feedback

through its multi-scale architecture

**Reference:** "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
Synthetic Data" (Wang et al., 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealESRGANVideo(NeuralNetworkArchitecture<>,RealESRGANVideoOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Real-ESRGAN Video model in native training mode. |
| `RealESRGANVideo(NeuralNetworkArchitecture<>,String,RealESRGANVideoOptions)` | Creates a Real-ESRGAN Video model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |


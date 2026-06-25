---
title: "PerVFI<T>"
description: "PerVFI: perception-oriented video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

PerVFI: perception-oriented video frame interpolation.

## For Beginners

PerVFI optimizes for what "looks good" rather than what's
mathematically closest, producing results that appear sharper and more natural.

**Usage:**

## How It Works

PerVFI (2024) uses perceptual quality optimization for frame interpolation:

- Perceptual loss hierarchy: uses a multi-scale perceptual loss computed from VGG/LPIPS

features, prioritizing visual quality over pixel-exact reconstruction

- Asymmetric distortion: applies different loss weights to different frequency bands,

penalizing structural errors more heavily than texture errors

- Perception-guided refinement: iteratively improves visual quality in regions where the

perceptual loss is highest using a refinement network with perceptual error maps

- Motion-aware perceptual attention: uses estimated motion magnitude to weight the perceptual

loss, applying stronger constraints in high-motion regions

**Reference:** "PerVFI: Perception-Oriented Video Frame Interpolation" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerVFI(NeuralNetworkArchitecture<>,PerVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a PerVFI model in native training mode. |
| `PerVFI(NeuralNetworkArchitecture<>,String,PerVFIOptions)` | Creates a PerVFI model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |


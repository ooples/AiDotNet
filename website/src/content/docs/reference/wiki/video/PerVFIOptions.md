---
title: "PerVFIOptions"
description: "Configuration options for PerVFI perception-oriented video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for PerVFI perception-oriented video frame interpolation.

## For Beginners

Most frame interpolation methods try to match pixels exactly, but
human eyes care more about visual quality than pixel accuracy. PerVFI optimizes for what
"looks good" rather than what's mathematically closest, producing results that appear
sharper and more natural even if pixel differences are slightly larger.

## How It Works

PerVFI (2024) uses perceptual quality optimization for frame interpolation:

- Perceptual loss hierarchy: instead of optimizing only pixel-level L1/L2 loss, PerVFI uses

a multi-scale perceptual loss computed from features of a pre-trained VGG/LPIPS network,
prioritizing visual quality over pixel-exact reconstruction

- Asymmetric distortion: applies different loss weights to different frequency bands,

penalizing low-frequency (structural) errors more heavily than high-frequency (texture)
errors, matching human visual perception priorities

- Perception-guided refinement: a refinement network that takes the initial interpolation

result plus a perceptual error map and iteratively improves visual quality in regions
where the perceptual loss is highest

- Motion-aware perceptual attention: uses estimated motion magnitude to weight the perceptual

loss, applying stronger perceptual constraints in high-motion regions where artifacts are
most visible

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerVFIOptions` | Initializes a new instance with default values. |
| `PerVFIOptions(PerVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFlowScales` | Gets or sets the number of flow estimation scales. |
| `NumRefinementIters` | Gets or sets the number of perceptual refinement iterations. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the refinement network. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PerceptualWeight` | Gets or sets the perceptual loss weight relative to reconstruction loss. |
| `Variant` | Gets or sets the model variant. |


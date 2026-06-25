---
title: "ToonCrafterOptions"
description: "Configuration options for ToonCrafter cartoon/anime video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for ToonCrafter cartoon/anime video frame interpolation.

## For Beginners

Regular frame interpolation is designed for real-world videos and
fails on cartoons/anime because animated characters move in exaggerated, non-realistic
ways. ToonCrafter is specially trained on animation content to produce natural-looking
in-between frames for cartoons.

## How It Works

ToonCrafter (2024) specializes in cartoon and anime frame interpolation:

- Cartoon-aware diffusion: adapts a video diffusion model specifically for cartoon/anime

content, where motion is typically non-physical (e.g., smear frames, anticipation poses)
and cannot be captured by standard optical flow methods

- Style-preserving generation: uses CLIP-based style conditioning to maintain consistent

art style, line weight, and coloring throughout the interpolated sequence

- Large motion handling: the diffusion backbone can generate plausible in-betweens even

for extreme pose changes common in hand-drawn animation

- Sketch-guided control: optional sketch/line art conditioning for artist control over

intermediate poses and motion trajectories

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToonCrafterOptions` | Initializes a new instance with default values. |
| `ToonCrafterOptions(ToonCrafterOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `GuidanceScale` | Gets or sets the guidance scale for classifier-free guidance. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion denoising steps. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumResBlocks` | Gets or sets the number of U-Net residual blocks per level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |


---
title: "StreamDiffVSROptions"
description: "Configuration options for the Stream-DiffVSR low-latency streaming video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the Stream-DiffVSR low-latency streaming video super-resolution model.

## For Beginners

Stream-DiffVSR is designed for live video upscaling where you can't
look at future frames. It uses a trick called "distillation" to reduce the number of
processing steps from ~50 to just 4, making it fast enough for real-time streaming.

## How It Works

Stream-DiffVSR (Li et al., 2025) achieves low-latency online video super-resolution through:

- Auto-regressive temporal guidance: uses previously generated HR frames to condition current denoising
- 4-step distilled denoiser: compresses many diffusion steps into just 4 for low latency
- Causal temporal conditioning: only looks at past frames, enabling streaming applications

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StreamDiffVSROptions` | Initializes a new instance with default values. |
| `StreamDiffVSROptions(StreamDiffVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LatentDim` | Gets or sets the latent space dimension for the diffusion process. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDenoisingSteps` | Gets or sets the number of denoising steps (distilled). |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the denoiser. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `TemporalRadius` | Gets or sets the temporal radius for causal conditioning (past frames only). |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps for the learning rate schedule. |


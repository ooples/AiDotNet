---
title: "ACEStepOptions"
description: "Configuration options for the ACE-Step model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the ACE-Step model.

## For Beginners

ACE-Step generates music from text descriptions super fast.
While most AI music generators need many steps (like painting layer by layer), ACE-Step
can create music in just 1-4 steps, making it fast enough for real-time use.

## How It Works

ACE-Step (2024) is an Accelerated Consistency-Enhanced music generation model that uses
consistency training to generate high-quality music in very few diffusion steps (1-4 steps
vs 50-100 for standard diffusion). It achieves real-time music generation while maintaining
quality comparable to multi-step models.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LatentDim` | Gets or sets the latent dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumChannels` | Gets or sets the number of output channels. |
| `NumSteps` | Gets or sets the number of inference steps. |
| `NumUNetLayers` | Gets or sets the number of U-Net layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the output sample rate in Hz. |
| `TextEncoderDim` | Gets or sets the text encoder dimension. |
| `UNetDim` | Gets or sets the U-Net hidden dimension. |
| `Variant` | Gets or sets the model variant. |


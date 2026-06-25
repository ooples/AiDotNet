---
title: "DynamiCrafterOptions"
description: "Configuration options for the DynamiCrafter video diffusion interpolation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the DynamiCrafter video diffusion interpolation model.

## For Beginners

DynamiCrafter uses an AI image/video generator (diffusion model)
that already knows how things move in the real world. Given a start frame and end frame,
it gradually "imagines" what happens in between, producing natural-looking intermediate
frames with realistic motion, lighting changes, and object interactions.

## How It Works

DynamiCrafter (2024) uses video diffusion priors for frame interpolation:

- Video diffusion backbone: adapts a pre-trained text-to-video diffusion model (e.g.,

Stable Video Diffusion) for the interpolation task, leveraging its learned motion priors

- First/last frame conditioning: the diffusion process is conditioned on both the first

and last frames using CLIP image embeddings injected via cross-attention, ensuring the
generated intermediate frames are temporally consistent with both endpoints

- Noise schedule adaptation: modified diffusion noise schedule that biases early denoising

steps toward global motion consistency and later steps toward local detail refinement

- Temporal attention: 3D self-attention across generated frames ensures smooth motion

transitions without flickering or temporal discontinuities

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynamiCrafterOptions` | Initializes a new instance with default values. |
| `DynamiCrafterOptions(DynamiCrafterOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion timesteps during inference. |
| `NumFeatures` | Gets or sets the number of feature channels in the UNet. |
| `NumHeads` | Gets or sets the number of attention heads in temporal attention. |
| `NumResBlocks` | Gets or sets the number of UNet residual blocks per level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |


---
title: "Wan21Model<T>"
description: "Wan 2.1 video model with MoE denoising and full 3D attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Wan 2.1 video model with MoE denoising and full 3D attention.

## For Beginners

Wan 2.1 is an open-source AI that creates videos from text descriptions
or still images. Think of it as a very advanced animation tool: you describe a scene in words, and
it generates a realistic video clip. It uses a technique called "Mixture of Experts" where different
parts of the model specialize in different aspects of video creation (lighting, motion, detail), then
combine their outputs. With 14 billion parameters, it produces some of the highest quality open-source
video generation results available.

## How It Works

**References:**

- Paper: "Wan: Open and Advanced Large-Scale Video Generative Models" (Alibaba, 2025)

Wan 2.1 introduces Mixture-of-Experts (MoE) into the DiT denoising backbone, enabling
specialized experts for different denoising phases. Combined with the Causal 3D VAE and flow matching
training, it achieves state-of-the-art video quality among open-source models. The model supports
both text-to-video and image-to-video generation with 14B parameters.

Technical specifications:

- Architecture: DiT + MoE + Causal3DVAE + FlowMatching
- Latent channels: 16
- Default: 81 frames at 16 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Wan21Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of Wan21Model with full customization support. |


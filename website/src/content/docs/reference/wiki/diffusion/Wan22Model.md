---
title: "Wan22Model<T>"
description: "Wan 2.2 video model with timestep-specialized MoE experts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Wan 2.2 video model with timestep-specialized MoE experts.

## For Beginners

Wan 2.2 improves on Wan 2.1 by assigning different expert sub-networks to different stages of the video creation process. Early stages focus on overall structure while later stages refine fine details. This timestep specialization produces sharper, more temporally consistent videos.

## How It Works

**References:**

- Paper: "Wan 2.2: Timestep-Specialized Mixture-of-Experts for Video Generation" (Alibaba, 2025)

Wan 2.2 upgrades the MoE architecture with timestep-specialized experts, where different
expert subnetworks activate at different denoising stages. Early timesteps use structure experts
while later timesteps use detail experts. This specialization significantly improves both
fidelity and temporal coherence.

Technical specifications:

- Architecture: DiT + Timestep-Specialized MoE + Causal3DVAE
- Latent channels: 16
- Default: 81 frames at 16 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Wan22Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of Wan22Model with full customization support. |


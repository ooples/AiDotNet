---
title: "FateZeroModel<T>"
description: "FateZero zero-shot video editing via attention blending."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.VideoEditing`

FateZero zero-shot video editing via attention blending.

## For Beginners

FateZero performs zero-shot video editing by blending attention patterns from the original and edited videos. It requires no training or fine-tuning - just describe the desired change and it applies it consistently across frames.

## How It Works

**References:**

- Paper: "FateZero: Fusing Attentions for Zero-shot Text-based Video Editing" (Qi et al., 2023)

FateZero enables zero-shot video editing by fusing self-attention maps from the source video's
DDIM inversion with cross-attention maps from the target prompt. This attention blending preserves
the original video's motion and structure while applying the desired edit. No per-video training
or optimization is required.

Technical specifications:

- Architecture: Attention Blending + DDIM Inversion + Zero-Shot Editing
- Latent channels: 4
- Default: 24 frames at 8 FPS
- Supports I2V: No | T2V: Yes | V2V: Yes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FateZeroModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of FateZeroModel with full customization support. |


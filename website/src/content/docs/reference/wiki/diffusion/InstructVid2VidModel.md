---
title: "InstructVid2VidModel<T>"
description: "InstructVid2Vid natural language instruction-based video editing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.VideoEditing`

InstructVid2Vid natural language instruction-based video editing.

## For Beginners

InstructVid2Vid allows natural language video editing - describe what you want changed (e.g., make it snowy) and the model applies that edit across all frames while preserving motion and structure.

## How It Works

**References:**

- Paper: "InstructVid2Vid: Controllable Video Editing with Natural Language Instructions" (ICME, 2024)

InstructVid2Vid enables video editing through natural language instructions, similar to
InstructPix2Pix but extended to video. The model learns to follow editing instructions while
maintaining temporal consistency across frames. Supports diverse edits including style transfer,
object manipulation, and scene modification.

Technical specifications:

- Architecture: Instruction-Conditioned UNet + Temporal Consistency
- Latent channels: 4
- Default: 24 frames at 8 FPS
- Supports I2V: No | T2V: Yes | V2V: Yes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstructVid2VidModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of InstructVid2VidModel with full customization support. |


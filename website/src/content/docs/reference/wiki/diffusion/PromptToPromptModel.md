---
title: "PromptToPromptModel<T>"
description: "Prompt-to-Prompt model for attention-based image editing by manipulating cross-attention maps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Prompt-to-Prompt model for attention-based image editing by manipulating cross-attention maps.

## For Beginners

Prompt-to-Prompt edits images by changing the prompt and controlling attention maps.

How Prompt-to-Prompt works:

1. Generate an image with the original prompt, storing cross-attention maps at each step
2. Modify the prompt (e.g., "a cat sitting" to "a dog sitting")
3. During re-generation, inject stored attention maps for unchanged words
4. Only attention maps for changed words are regenerated
5. This preserves the overall composition while editing specific elements

Editing modes:

- Word swap: "a cat sitting" to "a dog sitting" (replaces one element)
- Attention reweight: increase/decrease attention to specific words (make something bigger/smaller)
- Refinement: add detail to specific regions without changing structure

When to use Prompt-to-Prompt:

- Structure-preserving image edits via text changes
- Swapping objects while maintaining composition
- Adjusting emphasis on specific image elements
- Research on attention-based image control

Limitations:

- Requires deterministic scheduler (DDIM) for attention consistency
- Quality depends on attention map alignment between prompts
- Complex structural changes may break attention correspondence
- SD 1.5 resolution (512x512)

## How It Works

Prompt-to-Prompt enables image editing by directly manipulating the cross-attention maps
during the diffusion process. By controlling which attention maps are replaced, refined,
or reweighted between the original and edited prompts, users can make precise localized
edits to generated or real images without masks.

Architecture components:

- SD 1.5 U-Net backbone (320 base channels, [1,2,4,4], 768-dim CLIP)
- Cross-attention map extraction and manipulation during inference
- Three editing modes: word swap, attention reweight, refinement
- Standard SD 1.5 VAE for image encoding/decoding
- DDIM scheduler for deterministic attention control

Technical specifications:

- Architecture: SD 1.5 U-Net with cross-attention manipulation
- Backbone: SD 1.5 (320 base channels, [1,2,4,4] multipliers)
- Cross-attention: 768-dim (CLIP ViT-L/14)
- Editing modes: word-swap, attention-reweight, refinement
- Default resolution: 512x512
- Scheduler: DDIM (deterministic for attention consistency)

Reference: Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control", ICLR 2023

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |


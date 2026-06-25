---
title: "InstructPix2PixModel<T>"
description: "InstructPix2Pix model for instruction-based image editing via natural language text prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

InstructPix2Pix model for instruction-based image editing via natural language text prompts.

## For Beginners

InstructPix2Pix edits images using text instructions.

How InstructPix2Pix works:

1. Input image is encoded to latent space via VAE (4 channels)
2. Text instruction is encoded by CLIP into 768-dim features
3. Image latent is concatenated with noise latent (4+4 = 8 input channels)
4. U-Net denoises with dual guidance: image fidelity + instruction following
5. Image guidance scale controls how much to preserve the original image
6. Text guidance scale controls how strongly to follow the instruction
7. Decoded result is the edited image

Key characteristics:

- Natural language editing: "make the sky sunset colors", "add snow"
- No mask needed: the model figures out what to change
- Dual guidance: image guidance (1.0-2.0) + text guidance (7.0-12.0)
- Based on SD 1.5 with additional input channels
- Trained on GPT-4 generated instruction-image pairs

When to use InstructPix2Pix:

- Text-based image editing without masks
- Batch editing with consistent instructions
- Style transfer via natural language
- Quick creative edits from text descriptions

Limitations:

- Edit quality depends on instruction clarity
- May over-edit or under-edit without careful guidance tuning
- 512x512 base resolution (SD 1.5)
- Cannot handle fine-grained local edits as well as mask-based methods

## How It Works

InstructPix2Pix enables editing images by following natural language instructions.
It takes an input image and a text instruction (e.g., "make it winter") and produces
the edited result without requiring masks or per-example fine-tuning. The model uses
dual classifier-free guidance with separate image and text guidance scales.

Architecture components:

- SD 1.5 U-Net with 8 input channels (4 latent noise + 4 image conditioning)
- CLIP ViT-L/14 text encoder for 768-dim instruction embedding
- Dual classifier-free guidance (image guidance + text guidance)
- Standard SD 1.5 VAE for image encoding/decoding
- Euler discrete scheduler for efficient inference

Technical specifications:

- Architecture: SD 1.5 U-Net with 8 input channels
- Input: 8 channels (4 latent noise + 4 image conditioning)
- Text encoder: CLIP ViT-L/14 (768-dim)
- Image guidance scale: 1.0-2.0 recommended
- Text guidance scale: 7.0-12.0 recommended
- Default resolution: 512x512
- Scheduler: Euler discrete

Reference: Brooks et al., "InstructPix2Pix: Learning to Follow Image Editing Instructions", CVPR 2023

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `DefaultImageGuidance` | Gets the default image guidance scale for balancing image fidelity. |
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


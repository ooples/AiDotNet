---
title: "IPAdapterFaceIDModel<T>"
description: "IP-Adapter FaceID model for face-specific identity preservation using facial recognition embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

IP-Adapter FaceID model for face-specific identity preservation using facial recognition embeddings.

## For Beginners

IP-Adapter FaceID preserves face identity in generated images.

How IP-Adapter FaceID works:

1. A reference face image is processed by ArcFace to extract a 512-dim face embedding
2. The face embedding is projected to cross-attention token space via a learned projection
3. Face tokens are injected into the U-Net's cross-attention alongside text tokens
4. The U-Net generates an image that matches both the text prompt and reference face
5. Optional LoRA weights further improve generation quality and identity fidelity

Key differences from standard IP-Adapter:

- Uses face recognition (ArcFace) embeddings instead of CLIP image embeddings
- Much better face identity preservation (ID similarity)
- Specialized for face-centric generation tasks
- Compatible with SD 1.5 and SDXL backbones

When to use IP-Adapter FaceID:

- High-fidelity face identity transfer to new scenes
- Face-consistent character generation across images
- Personalized avatar and portrait creation
- Marketing and content with specific faces

Limitations:

- Requires clear frontal face in reference image
- May struggle with extreme poses or occlusions
- Face identity can drift with complex text prompts
- Single-face: multi-face requires separate handling

## How It Works

IP-Adapter FaceID extends the IP-Adapter framework with facial recognition embeddings
(ArcFace/InsightFace) instead of CLIP image embeddings, providing significantly more
accurate face identity preservation during image generation.

Architecture components:

- SD 1.5 U-Net backbone (320 base channels, [1,2,4,4], 768-dim CLIP)
- ArcFace/InsightFace face encoder producing 512-dim face embeddings
- Face embedding projection layer to cross-attention token space
- Optional LoRA layers for improved generation quality
- Standard SD 1.5 VAE for image encoding/decoding
- Euler discrete scheduler for efficient inference

Technical specifications:

- Architecture: IP-Adapter with ArcFace face encoder
- Face encoder: ArcFace/InsightFace (512-dim embedding)
- Backbone: SD 1.5 (320 base, [1,2,4,4], 768-dim CLIP)
- Projection: face embedding to cross-attention tokens
- Default resolution: 512x512
- Scheduler: Euler discrete
- Optional: LoRA for quality improvement
- Compatible: SD 1.5, SDXL

Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", 2023

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


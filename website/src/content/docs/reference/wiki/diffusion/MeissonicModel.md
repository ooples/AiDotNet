---
title: "MeissonicModel<T>"
description: "Meissonic model for non-autoregressive masked image modeling (MIM) generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Meissonic model for non-autoregressive masked image modeling (MIM) generation.

## For Beginners

Unlike diffusion models that gradually remove noise over many steps,
Meissonic works by masking parts of an image and predicting all missing parts at once.

How Meissonic works:

1. Text is encoded by a CLIP text encoder
2. Image tokens are randomly masked according to a cosine schedule
3. An E-MMDiT transformer predicts all masked tokens simultaneously
4. Iterative refinement: re-mask low-confidence tokens and predict again
5. VQ-VAE decodes discrete tokens to pixel space

Key characteristics:

- Non-autoregressive: predicts ALL masked tokens in parallel (not one by one)
- E-MMDiT backbone: only 304M parameters (very lightweight)
- Cosine masking schedule for iterative refinement
- CLIP text encoder for conditioning
- 16 latent channels via VQ-VAE tokenizer
- 10-20 refinement iterations (faster than 50-step diffusion)

Advantages:

- Very fast generation (10-20 iterations)
- Extremely lightweight (304M vs 2-12B for competitors)
- Runs on consumer GPUs with minimal VRAM
- High quality despite small size

Limitations:

- Quality below top diffusion models for complex scenes
- Discrete token space can show quantization artifacts
- Less fine-grained control than continuous latent diffusion

## How It Works

Meissonic uses masked image modeling (MIM) with a non-autoregressive approach for fast,
high-quality image generation. Instead of iterative denoising, it masks and predicts image
tokens in parallel using a lightweight E-MMDiT backbone (304M parameters), achieving fast
generation with quality competitive with much larger diffusion models.

Technical specifications:

- Architecture: Non-autoregressive masked generative transformer
- Backbone: E-MMDiT with 304M parameters
- Hidden size: 1024, 12 layers, 16 heads
- Text encoder: CLIP (768-dim)
- Tokenizer: VQ-VAE with 16 channels
- Masking: Cosine schedule with iterative refinement
- Default: 18 refinement iterations, guidance scale 9.0
- Resolution: 1024x1024

Reference: Bai et al., "Meissonic: Revitalizing Masked Generative Transformers
for Efficient High-Resolution Text-to-Image Synthesis", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeissonicModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,EMMDiTPredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of MeissonicModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `UsesMaskedModeling` | Gets whether this model uses masked image modeling (true) vs continuous diffusion. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for Meissonic (1024x1024). |
| `DefaultWidth` | Default image width for Meissonic (1024x1024). |


---
title: "SANAModel<T>"
description: "SANA model for efficient high-resolution text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

SANA model for efficient high-resolution text-to-image generation.

## For Beginners

SANA generates high-quality images very efficiently. While other
models need billions of parameters, SANA achieves similar quality with only 600 million,
making it much faster and requiring less memory.

How SANA works:

1. Text is encoded by a Gemma 2B language model (Google's text encoder)
2. A 20-layer linear DiT processes text embeddings and noise with O(n) attention
3. DC-AE with 32x spatial compression decodes tiny latents to full images
4. Flow matching training enables efficient 20-step generation

Model variants:

- SANA-0.6B: Default, 600M parameter DiT
- SANA-1.6B: Larger 1.6B parameter variant for higher quality

Key characteristics:

- 0.6B parameters in the transformer (vs 2-12B for competitors)
- 32x spatial compression via DC-AE (vs 8x standard)
- Linear attention: O(n) instead of O(n^2) quadratic attention
- Gemma 2B text encoder for strong multilingual prompt understanding
- 32 latent channels for high information retention
- Up to 4096x4096 resolution generation

Advantages:

- Extremely efficient: 100x+ faster than Flux1/SDXL for similar quality
- Low VRAM: runs on consumer GPUs with 8GB VRAM
- High resolution: native 4K generation support
- Strong text rendering via Gemma encoder

Limitations:

- Less community ecosystem than Stable Diffusion family
- Fewer LoRA/ControlNet adaptations available
- 32x compression can lose very fine details at low resolution

## How It Works

SANA uses a linear DiT architecture with efficient linear attention for high-resolution
generation (up to 4K). It achieves state-of-the-art quality with a compact 0.6B parameter
model through Deep Compression Autoencoder (DC-AE) with 32x spatial compression and
linear attention mechanisms that reduce compute from O(n^2) to O(n).

Technical specifications:

- Architecture: Linear DiT with efficient linear attention
- Transformer: 0.6B params, hidden 2240, 20 layers, 20 heads
- Text encoder: Gemma 2B (2048-dim embeddings)
- VAE: DC-AE with 32 latent channels, 32x spatial compression
- Training: Flow matching with linear noise schedule
- Default: 20 inference steps, guidance scale 4.5
- Resolution: 512x512 to 4096x4096 (aspect-ratio aware)

Reference: Xie et al., "SANA: Efficient High-Resolution Image Synthesis with
Linear Diffusion Transformers", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SANAModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,EMMDiTPredictor<>,DeepCompressionVAE<>,IConditioningModule<>,SANAVariant,Nullable<Int32>)` | Initializes a new instance of SANAModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressionFactor` | Gets the spatial compression factor of the DC-AE (32x). |
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |
| `Variant` | Gets the model variant. |

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
| `DefaultHeight` | Default image height for SANA (1024x1024). |
| `DefaultWidth` | Default image width for SANA (1024x1024). |


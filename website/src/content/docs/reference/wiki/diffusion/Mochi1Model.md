---
title: "Mochi1Model<T>"
description: "Mochi 1 model for asymmetric DiT video generation with joint text-video attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Mochi 1 model for asymmetric DiT video generation with joint text-video attention.

## For Beginners

Mochi 1 is a state-of-the-art open-source video generation model.

How Mochi 1 works:

1. Text prompt is encoded by T5-XXL into 4096-dim embeddings
2. Video is compressed by the asymmetric VAE into 12-channel latent space
3. The AsymmDiT processes text and video tokens jointly (not via cross-attention)
4. Joint attention allows deep text-video interaction in every layer
5. The heavy VAE decoder reconstructs high-quality video

Key characteristics:

- ~10B parameters (one of the largest open-source video models)
- Joint text-video attention (deeper integration than cross-attention)
- Asymmetric VAE: fast encoding, high-quality decoding
- 480p at 30 FPS, 84 frames (~2.8 seconds)
- Open-source weights under Apache 2.0 license

When to use Mochi 1:

- High-quality open-source video generation
- Research on joint attention mechanisms
- Text-to-video with strong motion understanding
- Commercial use (Apache 2.0 license)

Limitations:

- Very large model (10B parameters, high VRAM requirements)
- Limited to 480p resolution
- Shorter duration than some competitors
- Slower inference due to model size

## How It Works

Mochi 1 by Genmo uses an Asymmetric Diffusion Transformer (AsymmDiT) architecture
with joint text-video attention and an asymmetric encoder-decoder VAE.
The asymmetric design uses lightweight encoding and heavy decoding for quality.

Architecture components:

- AsymmDiT with 48 transformer layers and 3072 hidden dimension
- 24 attention heads with joint text-video attention (not cross-attention)
- Asymmetric VAE: lightweight encoder, heavy decoder for quality
- 12 latent channels with 3D causal temporal compression
- T5-XXL text encoder for 4096-dim context embeddings
- Flow matching training objective

Technical specifications:

- Architecture: AsymmDiT (Asymmetric Diffusion Transformer)
- Hidden dimension: 3072
- Transformer layers: 48
- Attention heads: 24
- Patch size: 2
- Latent channels: 12 (asymmetric VAE)
- Context dimension: 4096 (T5-XXL)
- Attention: Joint text-video (not cross-attention)
- Default: 84 frames at 30 FPS (~2.8 seconds)
- Training objective: Flow matching
- Total parameters: ~10B
- License: Apache 2.0

Reference: Genmo, "Mochi 1: A New SOTA in Open-Source Video Generation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mochi1Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of Mochi1Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsImageToVideo` |  |
| `SupportsTextToVideo` |  |
| `SupportsVideoToVideo` |  |
| `TemporalVAE` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(DiTNoisePredictor<>,TemporalVAE<>,Nullable<Int32>)` | Initializes the AsymmDiT and asymmetric VAE layers using custom or default configurations. |
| `PredictVideoNoise(Tensor<>,Int32,Tensor<>,Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CONTEXT_DIM` | Context dimension from the T5-XXL text encoder (4096). |
| `DEFAULT_FPS` | Default frames per second (30). |
| `DEFAULT_NUM_FRAMES` | Default number of frames (84, ~2.8 seconds at 30 FPS). |
| `HIDDEN_DIM` | Hidden dimension of the AsymmDiT transformer (3072). |
| `LATENT_CHANNELS` | Number of latent channels from the asymmetric VAE (12). |
| `NUM_HEADS` | Number of attention heads (24). |
| `NUM_LAYERS` | Number of transformer layers in the AsymmDiT (48). |
| `PATCH_SIZE` | Patch size for spatiotemporal tokenization (2). |
| `_conditioner` | The T5-XXL text encoder conditioning module. |
| `_dit` | The AsymmDiT noise predictor with joint text-video attention. |
| `_temporalVAE` | The asymmetric temporal VAE (lightweight encoder, heavy decoder). |


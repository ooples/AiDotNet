---
title: "PixArtModel<T>"
description: "PixArt-α model for efficient high-quality text-to-image generation using DiT architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

PixArt-α model for efficient high-quality text-to-image generation using DiT architecture.

## For Beginners

PixArt-α is like a sports car version of image generation:

Key advantages over traditional models:

- 10x faster training than Stable Diffusion
- Much more parameter-efficient
- Uses transformer blocks instead of U-Net
- T5-XXL text encoder for better prompt understanding

How PixArt-α works:

1. Your prompt goes through a T5-XXL text encoder (larger = better understanding)
2. The DiT (Diffusion Transformer) denoises using attention blocks
3. Each block uses cross-attention to the text embedding
4. The output is decoded by a VAE into an image

Example use cases:

- Fast prototyping (quick iterations)
- Resource-constrained environments (smaller models)
- High-quality generation without massive GPU requirements
- Applications requiring many generations

When to choose PixArt-α:

- You need faster generation than SDXL
- You want good quality without 70B+ model overhead
- Your prompts are complex (T5 encoder helps)
- You're doing many generations in batch

## How It Works

PixArt-α is an efficient text-to-image diffusion model that uses a Diffusion Transformer (DiT)
architecture. It achieves comparable quality to larger models like Stable Diffusion XL while
being significantly faster and more resource-efficient.

Technical specifications:

- Architecture: Diffusion Transformer (DiT) with AdaLN-single
- Text encoder: T5-XXL (4.3B parameters, optional smaller variants)
- Native resolutions: 256x256 to 1024x1024
- Latent space: 4 channels, 8x spatial downsampling
- Training: Decomposed training strategy for efficiency

Architecture innovations:

- Cross-attention in every DiT block
- AdaLN-single for timestep conditioning (not AdaLN-Zero)
- Efficient attention patterns
- Multi-resolution training support

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PixArtModel(NeuralNetworkArchitecture<>,PixArtVariant,IConditioningModule<>,INoiseScheduler<>,Nullable<Int32>)` | Initializes a new instance of PixArtModel with default parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `DefaultResolution` | Gets the default resolution for this model. |
| `HiddenDimension` | Gets the hidden dimension of the transformer. |
| `LatentChannels` |  |
| `ModelSize` | Gets the model variant. |
| `NoisePredictor` |  |
| `NumAttentionHeads` | Gets the number of attention heads. |
| `NumLayers` | Gets the number of transformer layers. |
| `ParameterCount` |  |
| `SupportsVariableAspectRatio` | Gets whether this model supports variable aspect ratios. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDimensionsForAspectRatio(String,Int32)` | Calculates width and height for a given aspect ratio. |
| `Clone` |  |
| `CreateDefaultOptions` | Creates the default options for PixArt-α. |
| `CreateDefaultScheduler(Nullable<Int32>)` | Creates the default scheduler for PixArt-α. |
| `CreateDiTPredictor(Nullable<Int32>)` | Creates a DiT predictor configured for PixArt-α. |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image with PixArt-α's efficient DiT architecture. |
| `GenerateVariations(String,String,Int32,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates multiple image variations with different seeds. |
| `GenerateWithAspectRatio(String,String,String,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image with specified aspect ratio preset. |
| `GetModelConfiguration(PixArtVariant)` | Gets model configuration based on size variant. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `GetRecommendedSettings` | Gets the recommended settings for this model variant. |
| `GetSupportedResolutions` | Gets supported resolutions for this model variant. |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs image-to-image transformation with PixArt-α. |
| `InitializeLayers(Nullable<Int32>)` | Initializes the DiT noise predictor and VAE layers. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultVariant` | Default model variant. |
| `PIXART_LATENT_CHANNELS` | Standard PixArt latent channels. |
| `PIXART_VAE_SCALE_FACTOR` | Standard VAE scale factor. |
| `_conditioner` | The conditioning module (T5-style text encoder). |
| `_defaultResolution` | Default resolution for this model variant. |
| `_dit` | The DiT-based noise predictor. |
| `_hiddenDim` | Hidden dimension for the transformer. |
| `_modelSize` | Model variant (Alpha, Sigma, Delta, XL). |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer blocks. |
| `_vae` | The VAE for encoding/decoding. |


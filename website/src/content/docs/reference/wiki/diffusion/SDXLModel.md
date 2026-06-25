---
title: "SDXLModel<T>"
description: "Stable Diffusion XL (SDXL) model for high-resolution image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Stable Diffusion XL (SDXL) model for high-resolution image generation.

## For Beginners

SDXL is like Stable Diffusion 2.0 but significantly upgraded:

Key improvements over SD 1.5/2.0:

- 4x larger U-Net (2.6B vs 865M parameters)
- Dual text encoders (better prompt understanding)
- Native 1024x1024 resolution (vs 512x512)
- Optional refiner model for enhanced details

How SDXL works:

1. Your prompt goes through TWO text encoders (CLIP + OpenCLIP)
2. These embeddings guide a much larger U-Net during denoising
3. The base model generates at 1024x1024
4. (Optional) A refiner model enhances fine details

Example prompt flow:
"A majestic dragon" -> [CLIP] + [OpenCLIP] -> Combined embedding
-> Large U-Net denoises -> 1024x1024 image
-> (Optional) Refiner -> Enhanced details

Use SDXL when you need:

- High resolution output
- Better text rendering in images
- More detailed and coherent images
- Following complex prompts accurately

## How It Works

SDXL is Stability AI's flagship text-to-image model, designed for
high-quality 1024x1024 image generation with improved prompt understanding
and visual fidelity compared to earlier Stable Diffusion versions.

Technical specifications:

- Base model: 2.6B parameter U-Net
- Text encoders: CLIP ViT-L/14 + OpenCLIP ViT-bigG/14
- Native resolution: 1024x1024
- Latent space: 4 channels, 8x spatial downsampling
- Guidance scale: 5.0-9.0 recommended (7.5 default)
- Scheduler: DDPM/DPM++/Euler with 20-50 steps

Architecture details:

- Micro-conditioning: Size and crop coordinates for multi-aspect training
- Dual text encoding: Concatenated CLIP + OpenCLIP embeddings
- Channel multipliers: [1, 2, 4, 4] (vs [1, 2, 4, 8] in SD 2.x)
- Cross-attention dimension: 2048 (vs 1024 in SD 1.x)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SDXLModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,IConditioningModule<>,SDXLRefiner<>,Boolean,Int32,Nullable<Int32>)` | Initializes a new instance of SDXLModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (2048 for SDXL). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `Refiner` | Gets the refiner model if available. |
| `SecondaryConditioner` | Gets the secondary text encoder if available. |
| `SupportsRefiner` | Gets whether this model has a refiner available. |
| `UsesDualEncoder` | Gets whether this model uses dual text encoders. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyMicroCondition(Tensor<>,Tensor<>)` | Concatenates micro-conditioning to text embedding. |
| `Clone` |  |
| `ConcatenateEmbeddings(Tensor<>,Tensor<>)` | Concatenates embeddings from two text encoders. |
| `CreateMicroCondition(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates micro-conditioning vector for aspect ratio handling. |
| `DeepCopy` |  |
| `EncodeTextDual(String)` | Encodes text using dual text encoders. |
| `EncodeTextDualAsync(String,CancellationToken)` | Async dual-encoder text encode. |
| `EnumerateDisposableComponents` |  |
| `GenerateAsync(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>,IProgress<Int32>,CancellationToken)` | Async wrapper around `Int32})` with cooperative cancellation between scheduler steps and per-step progress reporting. |
| `GenerateWithMicroCondition(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image with micro-conditioning for multi-aspect ratio support. |
| `GenerateWithMicroConditionCancellable(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>,IProgress<Int32>,CancellationToken)` | Cancellable variant of `Int32})` shared between the sync and async surfaces. |
| `GenerateWithMicroConditionTrulyAsync(String,String,Int32,Int32,Nullable<Int32>,Nullable<Int32>,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>,IProgress<Int32>,CancellationToken)` | True-async denoising path. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `GetUnconditionalEmbeddingDual` | Gets unconditional embedding for CFG with dual encoders. |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers. |
| `InvalidateAllStageCompiledPlans` | Drops every stage's plan in lockstep. |
| `InvalidateConditioner1CompiledPlans` | Drops the conditioner1 stage's compiled plan on the next generation. |
| `InvalidateConditioner2CompiledPlans` | Drops conditioner2's plan only. |
| `InvalidateUNetCompiledPlans` | Drops UNet's per-step plan only. |
| `InvalidateVAECompiledPlans` | Drops VAE-decoder's plan only. |
| `RefineImage(Tensor<>,String,String,Int32,Double,Nullable<Int32>)` | Refines an image using the SDXL refiner model. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default height for SDXL generation. |
| `DefaultWidth` | Default width for SDXL generation. |
| `SDXL_CROSS_ATTENTION_DIM` | Default cross-attention dimension for SDXL. |
| `SDXL_LATENT_CHANNELS` | Standard SDXL latent channels. |
| `SDXL_VAE_SCALE_FACTOR` | Standard SDXL VAE scale factor. |
| `_conditioner1` | The primary conditioning module (CLIP ViT-L). |
| `_conditioner1Version` | Per-stage structure-version stamps. |
| `_conditioner2` | The secondary conditioning module (OpenCLIP ViT-bigG). |
| `_crossAttentionDim` | Cross-attention dimension for SDXL (2048). |
| `_generationGate` | The U-Net noise predictor. |
| `_refiner` | Optional refiner model. |
| `_stageChain` | Composite chain that owns per-stage compile hosts for SDXL's {conditioner1, conditioner2, unet, vae-decode} pipeline. |
| `_useDualEncoder` | Whether to use dual text encoders. |
| `_vae` | The VAE for encoding/decoding. |


---
title: "LatentDiffusionModelBase<T>"
description: "Base class for latent diffusion models that operate in a compressed latent space."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion`

Base class for latent diffusion models that operate in a compressed latent space.

## For Beginners

This is the foundation for latent diffusion models like Stable Diffusion.
It combines a VAE (for compression), a noise predictor (for denoising), and optional conditioning
(for guided generation from text or images).

## How It Works

This abstract base class provides common functionality for all latent diffusion models,
including encoding/decoding, text-to-image generation, image-to-image transformation, and inpainting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LatentDiffusionModelBase(DiffusionModelOptions<>,INoiseScheduler<>,NeuralNetworkArchitecture<>)` | Initializes a new instance of the LatentDiffusionModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `GuidanceScale` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `SupportsInpainting` |  |
| `SupportsNegativePrompt` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGuidance(Tensor<>,Tensor<>,Double)` | Applies classifier-free guidance to combine conditional and unconditional predictions. |
| `BlendLatentsWithMask(Tensor<>,Tensor<>,Tensor<>,Int32)` | Blends generated latents with original latents based on mask for inpainting. |
| `CanonicalizeGenShape(Int32[],INoisePredictor<>)` | Maps a user-supplied input shape to the canonical generation shape that `predictor` can consume, preserving batch and spatial dims from the input. |
| `DecodeFromLatent(Tensor<>)` |  |
| `EncodeToLatent(Tensor<>,Boolean)` |  |
| `EnsureLatentShape(Tensor<>)` | Ensures a tensor has proper 4D latent shape [B, C, H, W] for UNet processing. |
| `Generate(Int32[],Int32,Nullable<Int32>)` |  |
| `GenerateAsync(Int32[],Int32,Nullable<Int32>,CancellationToken)` | Async overload of `Int32})` for latent diffusion (text encoder → noise predictor → VAE decode pipeline). |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetParameterChunks` | Streams the network's trainable weight tensors per-tensor without materialising a flat aggregate, mirroring PyTorch's `nn.Module.parameters()` generator pattern. |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `Inpaint(Tensor<>,Tensor<>,String,String,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `Predict(Tensor<>)` |  |
| `PredictNoise(Tensor<>,Int32)` |  |
| `ResizeMaskToLatent(Tensor<>,Int32[])` | Resizes a mask tensor to match latent dimensions. |
| `ResolveLatentShape(Int32[])` | Translates a caller-supplied `shape` into a latent-space shape suitable for the noise predictor's denoising loop. |
| `SampleNoiseTensor(Int32[],Random)` | Samples a noise tensor from standard normal distribution. |
| `SanitizeFiniteInPlace(Tensor<>)` | Replaces every NaN / Infinity element of `tensor` with zero. |
| `SetGuidanceScale(Double)` |  |
| `SetParameterChunks(IEnumerable<Tensor<>>)` | Streaming counterpart to `SetParameters`: distributes per-tensor chunks to the noise predictor, then the VAE, then the conditioner — the SAME order `GetParameterChunks` yields them — without materializing a flat aggregate. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_guidanceScale` | The default guidance scale for classifier-free guidance. |


---
title: "DallE3Model<T>"
description: "DALL-E 3 style text-to-image generation model with advanced prompt understanding and high-fidelity image generation capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

DALL-E 3 style text-to-image generation model with advanced prompt understanding
and high-fidelity image generation capabilities.

## For Beginners

DALL-E 3 generates high-quality images from text descriptions.
Unlike earlier versions, it deeply understands complex prompts with spatial relationships,
text rendering, and artistic styles. It uses a diffusion process (gradually refining random
noise into an image) combined with advanced prompt understanding to produce images that
closely match what you describe in words.

## How It Works

This implementation provides DALL-E 3 style capabilities including prompt expansion,
text rendering, style control, and high-quality image generation at multiple sizes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DallE3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of the DallE3Model class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `MaxPromptLength` |  |
| `NoisePredictor` |  |
| `ParameterCount` | Counts the flat-API parameter surface (predictor + VAE). |
| `SupportedSizes` |  |
| `SupportsEditing` |  |
| `SupportsVariations` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckPromptSafety(String)` |  |
| `Clone` |  |
| `CreateVariations(Tensor<>,Int32,Double,DallE3ImageSize)` |  |
| `DeepCopy` |  |
| `Edit(Tensor<>,Tensor<>,String,DallE3ImageSize)` |  |
| `EstimateQuality(String)` |  |
| `ExpandPrompt(String,DallE3Style)` |  |
| `Generate(String,DallE3ImageSize,DallE3Quality,DallE3Style,Nullable<Int32>)` |  |
| `GenerateConsistentSet(String,IEnumerable<String>,Int32,DallE3ImageSize)` |  |
| `GenerateForUseCase(String,String,DallE3ImageSize)` |  |
| `GenerateMultiple(String,Int32,DallE3ImageSize,DallE3Quality,DallE3Style)` |  |
| `GenerateTileable(String,DallE3ImageSize)` |  |
| `GenerateWithComposition(String,IEnumerable<ValueTuple<String,String,Double>>,DallE3ImageSize)` |  |
| `GenerateWithPrompt(String,DallE3ImageSize,DallE3Quality,DallE3Style)` |  |
| `GenerateWithStyle(String,String,DallE3ImageSize,DallE3Quality)` |  |
| `GenerateWithText(String,String,String,DallE3ImageSize)` |  |
| `GetModelMetadata` |  |
| `GetNextRandomSeed` | Gets a thread-safe random seed using RandomHelper. |
| `GetParameters` |  |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>)` | Initializes the model layers. |
| `Outpaint(Tensor<>,String,Int32,String)` |  |
| `SetParameters(Vector<>)` |  |
| `Upscale(Tensor<>,Int32,Boolean)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `LATENT_CHANNELS` |  |
| `MAX_PROMPT_LENGTH` |  |
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |
| `STANDARD_SIZE` |  |
| `TALL_HEIGHT` |  |
| `VAE_SCALE_FACTOR` |  |
| `WIDE_WIDTH` |  |


---
title: "IPAdapterModel<T>"
description: "IP-Adapter model for image-based prompt conditioning in diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

IP-Adapter model for image-based prompt conditioning in diffusion models.

## For Beginners

IP-Adapter lets you use pictures as instructions
for the AI instead of just text.

Think of it like:

- Showing someone a photo and saying "make something like this"
- The AI extracts the style, composition, and content from your image
- It then applies those elements to create new images

Use cases:

- Style transfer: "Generate in the style of this artwork"
- Face preservation: Keep a person's likeness in different scenes
- Object consistency: Maintain the same object across images
- Scene composition: Use reference for layout/arrangement

Key advantage: Combines with text prompts for precise control

## How It Works

IP-Adapter (Image Prompt Adapter) enables using reference images as prompts
to guide image generation. It decouples cross-attention for text and image
features, allowing fine-grained control over image style, composition, and
content transfer.

Technical details:

- Uses a pretrained image encoder (like CLIP ViT)
- Projects image features to text embedding space
- Injects via decoupled cross-attention mechanism
- Supports multiple reference images (multi-IP)
- Adjustable image prompt weight (0-1)

Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IPAdapterModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Nullable<Int32>)` | Initializes a new instance of IPAdapterModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `ImagePromptWeight` | Gets or sets the default image prompt weight (0-1). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `Clone` |  |
| `CombineEmbeddings(Tensor<>,Tensor<>)` | Combines text and image embeddings. |
| `DeepCopy` |  |
| `GenerateWithEmbedding(Tensor<>,Tensor<>,Int32,Int32,Int32,Double,Nullable<Int32>)` | Generates image using pre-computed embeddings. |
| `GenerateWithImagePrompt(String,Tensor<>,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Double>,Nullable<Int32>)` | Generates an image with image prompt conditioning. |
| `GenerateWithMultiImagePrompt(String,Tensor<>[],Double[],String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image with multiple image prompts. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ScaleTensor(Tensor<>,Double)` | Scales a tensor by a scalar value. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CLIP_EMBED_DIM` | CLIP image embedding dimension. |
| `IPA_LATENT_CHANNELS` | Standard IP-Adapter latent channels. |
| `IPA_VAE_SCALE_FACTOR` | Standard VAE scale factor. |
| `_baseUNet` | The base noise predictor (U-Net). |
| `_conditioner` | The text conditioning module. |
| `_imageEncoder` | The image encoder for extracting image features. |
| `_imageProjector` | The image projection layer. |
| `_imagePromptWeight` | Default image prompt weight. |
| `_vae` | The VAE for encoding/decoding. |


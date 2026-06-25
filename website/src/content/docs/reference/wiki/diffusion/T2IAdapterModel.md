---
title: "T2IAdapterModel<T>"
description: "T2I-Adapter model for adding spatial control to text-to-image diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

T2I-Adapter model for adding spatial control to text-to-image diffusion models.

## For Beginners

T2I-Adapter adds structural guidance to image generation:

How T2I-Adapter works:

1. A spatial condition (depth map, sketch, pose, etc.) is processed by the adapter network
2. The adapter produces multi-scale feature maps matching the U-Net encoder stages
3. These features are added to the U-Net during denoising for structural guidance
4. The base SD model generates images following both text AND spatial conditions

Key characteristics:

- Lightweight: Only ~77M parameters (vs ~860M for ControlNet)
- Pluggable: Works with any SD 1.5 or SDXL base model
- Composable: Multiple adapters can be combined (e.g., depth + sketch)
- No base model modification: Adapter weights are separate
- Fast training: Much faster to train than ControlNet

Supported condition types:

- Depth maps (MiDaS, ZoeDepth)
- Canny edge maps
- Sketch/line art
- OpenPose skeleton
- Semantic segmentation
- Color palette

Advantages over ControlNet:

- 10x fewer parameters
- Faster training and inference
- Composable (stack multiple adapters)
- Smaller model files

Limitations:

- Less precise control than ControlNet
- May not preserve fine spatial details as well

## How It Works

T2I-Adapter is a lightweight adapter architecture that adds spatial conditioning
(depth maps, sketches, pose, segmentation, etc.) to pre-trained text-to-image models
without modifying the base model weights.

Technical specifications:

- Architecture: Lightweight encoder adapter with multi-scale feature injection
- Adapter: ~77M parameters, 4 downsampling stages matching U-Net encoder
- Compatible base models: SD 1.5 (768-dim), SDXL (varies)
- Input conditions: Any spatial map (depth, edge, pose, etc.)
- Adapter scale: Adjustable strength [0.0, 1.0] for blending
- Adapter channels: [320, 640, 1280, 1280] matching SD U-Net

Reference: Mou et al., "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models", AAAI 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `T2IAdapterModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Double,Nullable<Int32>)` | Initializes a new instance of T2IAdapterModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterNetwork` | Gets the adapter network that processes spatial conditions. |
| `AdapterScale` | Gets the adapter conditioning scale [0.0, 1.0]. |
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
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `InitializeLayers(UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the base U-Net, adapter network, and VAE layers, using custom layers from the user if provided or creating industry-standard layers from the T2I-Adapter paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `ADAPTER_CROSS_ATTENTION_DIM` | Cross-attention dimension matching the base SD 1.5 model (768). |
| `ADAPTER_DEFAULT_CONDITIONING_SCALE` | Default adapter conditioning scale (1.0 = full strength). |
| `ADAPTER_DEFAULT_GUIDANCE_SCALE` | Default guidance scale (7.5, same as SD 1.5). |
| `DefaultHeight` | Default image height for T2I-Adapter (matches SD 1.5). |
| `DefaultWidth` | Default image width for T2I-Adapter (matches SD 1.5). |


---
title: "IPAdapterPlusModel<T>"
description: "IP-Adapter Plus model for image prompt conditioning in diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

IP-Adapter Plus model for image prompt conditioning in diffusion models.

## For Beginners

Instead of describing what you want with text, you can show
the AI a reference image. IP-Adapter Plus extracts the style and content from your
reference and applies them to the generation, like saying "make something like this."

## How It Works

IP-Adapter Plus enables image-based conditioning for diffusion models by extracting
image features through a vision encoder and injecting them via cross-attention.
The "Plus" variant uses fine-grained image features with decoupled cross-attention
for higher fidelity image prompting.

Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IPAdapterPlusModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Double,Nullable<Int32>)` | Initializes a new IP-Adapter Plus model. |

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
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |


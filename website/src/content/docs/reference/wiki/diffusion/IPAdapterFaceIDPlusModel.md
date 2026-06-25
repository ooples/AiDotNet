---
title: "IPAdapterFaceIDPlusModel<T>"
description: "IP-Adapter FaceID Plus model for face-identity-preserving generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

IP-Adapter FaceID Plus model for face-identity-preserving generation.

## For Beginners

Give this model a photo of someone's face, and it will
generate new images that look like the same person in different poses, styles,
or settings. It preserves the person's identity using face recognition technology.

## How It Works

Combines IP-Adapter's image prompting with face recognition embeddings
(FaceID) for identity-preserving face generation. Supports both face swapping
and face-consistent character creation.

Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IPAdapterFaceIDPlusModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Double,Nullable<Int32>)` | Initializes a new IP-Adapter FaceID Plus model. |

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


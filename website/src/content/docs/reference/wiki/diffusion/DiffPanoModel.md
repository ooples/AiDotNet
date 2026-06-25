---
title: "DiffPanoModel<T>"
description: "DiffPano model for scalable panorama generation with spherical epipolar-aware diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Panorama`

DiffPano model for scalable panorama generation with spherical epipolar-aware diffusion.

## For Beginners

DiffPano creates full 360-degree panoramic images that look consistent
from every viewing angle. It understands the geometry of spherical images (like what you see
in a VR headset) and ensures that all parts of the panorama connect seamlessly, even when
generating from just a single photo.

## How It Works

DiffPano uses spherical epipolar constraints to generate consistent 360-degree panoramic
images from single or multi-view inputs. It encodes geometric relationships between panoramic
viewpoints using epipolar-aware cross-attention, ensuring global consistency across the
full spherical field of view.

Reference: Wang et al., "DiffPano: Scalable and Consistent Text to Panorama Generation
with Spherical Epipolar-Aware Diffusion", NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffPanoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new DiffPano model with optional configuration. |

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


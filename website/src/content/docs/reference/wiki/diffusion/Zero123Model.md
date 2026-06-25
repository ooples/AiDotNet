---
title: "Zero123Model<T>"
description: "Zero-1-to-3 model for novel view synthesis from a single image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Zero-1-to-3 model for novel view synthesis from a single image.

## For Beginners

Zero123 is like having a magical camera that can show you
what an object looks like from different angles, even though you only gave it
one photograph.

What it does:

- Takes a single image of an object
- Generates images of that same object from different viewpoints
- Works with any object: cars, furniture, animals, etc.

Input parameters:

- Image: The original photo of the object
- Camera rotation: How much to rotate the view (polar/azimuth angles)
- Scale change: How close/far to zoom

Use cases:

- E-commerce: Show products from multiple angles
- 3D reconstruction: Generate training data for 3D models
- AR/VR: Create object previews from any angle
- Game development: Generate sprite variations

## How It Works

Zero-1-to-3 (Zero123) generates new viewpoints of an object from just a single
input image. It uses camera pose conditioning to control the viewpoint change,
enabling 3D-aware image generation without explicit 3D reconstruction.

Technical details:

- Fine-tuned from Stable Diffusion
- Uses CLIP image encoder for conditioning
- Camera pose embedding via sinusoidal encoding
- Supports arbitrary viewpoint changes
- Can be used iteratively for 360° reconstruction

Reference: Liu et al., "Zero-1-to-3: Zero-shot One Image to 3D Object", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Zero123Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,Int32,Nullable<Int32>)` | Initializes a new instance of Zero123Model with full customization support. |

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
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `Clone` |  |
| `CombineEmbeddings(Tensor<>,Tensor<>)` | Combines image and pose embeddings. |
| `ConcatenateLatents(Tensor<>,Tensor<>)` | Concatenates two latent tensors along the channel dimension. |
| `DeepCopy` |  |
| `Generate360Views(Tensor<>,Int32,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Generates multiple views around an object (360° views). |
| `GenerateMultipleViews(Tensor<>,Double[],Double[],Int32,Nullable<Double>,Nullable<Int32>)` | Generates views at multiple elevation angles. |
| `GenerateNovelView(Tensor<>,Double,Double,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Generates a novel view of an object. |
| `GetParameters` |  |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net, VAE, image encoder, and camera pose encoder. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_IMAGE_SIZE` | Default image size for Zero123 generation. |
| `Z123_LATENT_CHANNELS` | Number of latent channels in the VAE representation. |
| `Z123_VAE_SCALE_FACTOR` | Spatial downsampling factor of the VAE. |
| `_imageEncoder` | The image encoder for conditioning. |
| `_poseEncoder` | The camera pose encoder. |
| `_unet` | The U-Net noise predictor. |
| `_vae` | The VAE for encoding/decoding. |


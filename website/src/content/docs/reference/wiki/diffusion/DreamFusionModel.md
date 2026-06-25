---
title: "DreamFusionModel<T>"
description: "DreamFusion model for text-to-3D generation via Score Distillation Sampling (SDS)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

DreamFusion model for text-to-3D generation via Score Distillation Sampling (SDS).
Uses a 2D diffusion prior to optimize a 3D neural radiance field representation.
Based on "DreamFusion: Text-to-3D using 2D Diffusion" (Poole et al., 2022).

## For Beginners

Think of DreamFusion as using an AI art critic (the 2D diffusion model)
to guide a 3D sculptor (the NeRF). The critic looks at 2D views of the sculpture and gives
feedback on how to make it look more like the text description.

How it works:

1. You describe what you want: "a DSLR photo of a peacock on a surfboard"
2. DreamFusion renders 2D images of a 3D shape from random viewpoints
3. The 2D diffusion model evaluates: "Does this look like the prompt?"
4. Gradients flow back to improve the 3D representation
5. After many iterations, you get a full 3D object you can view from any angle

Key features:

- Creates full 3D assets from text descriptions
- View-consistent: looks correct from any angle
- Leverages the quality of 2D image generators
- No 3D training data required

## How It Works

DreamFusion revolutionized text-to-3D generation by using pretrained 2D diffusion models
to guide the optimization of a 3D scene representation (NeRF). The key insight is that
a 2D diffusion model can serve as a "critic" for 3D content through Score Distillation Sampling.

Technical details:

- Uses NeRF (Neural Radiance Field) for 3D representation
- Employs Score Distillation Sampling (SDS) loss
- Samples random camera views during optimization
- Uses classifier-free guidance with high scale (typically 100)
- Supports mesh extraction via marching cubes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DreamFusionModel(NeuralNetworkArchitecture<>,IDiffusionModel<>,DreamFusionConfig,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of the DreamFusionModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `Config` | Configuration for DreamFusion model. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoiseToImage(Tensor<>,Tensor<>,Int32)` | Adds noise to an image at a specific timestep. |
| `ApplyGuidanceInternal(Tensor<>,Tensor<>,Double)` | Applies classifier-free guidance to noise predictions (internal version). |
| `Clone` |  |
| `ComputeSDSGradient(Tensor<>,Tensor<>,Int32)` | Computes the Score Distillation Sampling gradient. |
| `ComputeSDSWeight(Int32)` | Computes the SDS weight for a timestep. |
| `DeepCopy` |  |
| `EncodeTextInternal(String)` | Encodes text to embedding. |
| `ExtractMesh(NeRFResult<>,Int32,Double)` | Generates a mesh from the trained NeRF using marching cubes. |
| `ExtractSurface(Double[0:,0:,0:],Double,Double,List<DreamVector3<>>,List<Int32>)` | Extracts surface using simplified marching cubes. |
| `GenerateAsync(String,Int32,Double,Double,IProgress<Double>,CancellationToken)` | Generates a 3D representation from a text prompt using Score Distillation Sampling. |
| `GenerateNoise(Tensor<>)` | Generates noise matching the input tensor dimensions. |
| `GetAlphasCumprod(Int32)` | Gets the cumulative alpha value for a timestep. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `GetSigma(Int32)` | Gets the noise level (sigma) for a timestep. |
| `InitializeLayers(Nullable<Int32>)` | Initializes the U-Net, VAE, and NeRF network layers. |
| `MarchingCubes(NeRFNetwork<>,Int32,Double)` | Extracts a mesh using marching cubes algorithm. |
| `PredictNoise(Tensor<>,Int32)` |  |
| `PredictNoiseWithEmbedding(Tensor<>,Int32,Tensor<>)` | Predicts noise using the diffusion prior with text embedding. |
| `RenderView(NeRFResult<>,CameraPose,Int32)` | Renders an image from the trained NeRF at a specific camera pose. |
| `SampleCameraPose` | Samples a random camera pose for rendering. |
| `SampleSDSTimestep` | Samples a timestep for SDS, favoring mid-range timesteps. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DEFAULT_BETA_END` | Default beta end value. |
| `DEFAULT_BETA_START` | Default beta start value. |
| `DEFAULT_TIMESTEPS` | Default timesteps for diffusion. |
| `DREAM_LATENT_CHANNELS` | Standard latent channels (same as SD). |


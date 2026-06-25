---
title: "ConsistencyModel<T>"
description: "Consistency Model for single-step or few-step image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Consistency Model for single-step or few-step image generation.

## For Beginners

Traditional diffusion models are like walking down a path one step
at a time - they need many steps to go from noise to a clear image. Consistency Models
are like teleportation - they can jump directly from any point on the path to the
destination (clean image).

Key advantages:

- Single-step generation possible (fastest mode)
- Progressive refinement: 1-step, 2-step, 4-step, etc.
- Quality improves with more steps but plateaus quickly
- Same or better quality as DDPM with 1000x fewer steps

How it works:

1. The model learns that all points on a denoising path should map to the same clean image
2. This "consistency" property allows direct prediction from any noise level
3. The model can self-refine by treating its output as a new starting point

Use cases:

- Real-time image generation
- Interactive applications
- Mobile/edge deployment
- Batch processing at scale

## How It Works

Consistency Models can generate high-quality images in a single step by learning to map
any point on a probability flow ODE trajectory directly to the trajectory's origin
(the clean data). This enables extremely fast generation compared to traditional
diffusion models that require 20-50+ steps.

Technical details:

- Based on probability flow ODEs (deterministic diffusion)
- Two training methods: distillation from pretrained diffusion, direct training
- Uses boundary condition f(x, eps) = x (identity at minimal noise)
- Supports multistep sampling for quality/speed tradeoff

Reference: Song et al., "Consistency Models", ICML 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConsistencyModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Double,Double,Double,Boolean,Nullable<Int32>)` | Initializes a new instance of ConsistencyModel with default parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsDistilled` | Gets whether this model was trained via distillation. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SigmaMax` | Gets the maximum sigma value used by this model. |
| `SigmaMin` | Gets the minimum sigma value used by this model. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeSigmaSchedule` | Computes the sigma schedule for the ODE. |
| `ComputeStepIndices(Int32)` | Computes the step indices for multistep sampling. |
| `ConsistencyFunction(Tensor<>,,Tensor<>)` | Applies the consistency function to map from any noise level to clean data. |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image using consistency sampling. |
| `GenerateWithProgressiveRefinement(String,String,Int32,Int32,Int32,Boolean,Nullable<Double>,Nullable<Int32>)` | Generates images with progressive refinement, optionally returning intermediates. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the noise predictor (eager) and the lazy-VAE factory. |
| `ScaleTensor(Tensor<>,Double)` | Scales a tensor by a scalar value. |
| `SetParameters(Vector<>)` |  |
| `SigmaToTimestep(Double)` | Converts sigma to timestep for the noise predictor. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CM_LATENT_CHANNELS` | Number of latent channels in the VAE representation. |
| `CM_VAE_SCALE_FACTOR` | Spatial downsampling factor of the VAE. |
| `_conditioner` | The conditioning module for text encoding. |
| `_isDistilled` | Whether this model was trained via distillation. |
| `_noisePredictor` | The noise predictor (U-Net or DiT). |
| `_numTrainSteps` | Number of training timesteps. |
| `_rho` | Rho parameter for sigma schedule. |
| `_sigmaMax` | Maximum sigma value. |
| `_sigmaMin` | Minimum sigma value (epsilon). |
| `_sigmas` | The sigma schedule for the ODE. |
| `_vae` | The VAE for encoding/decoding. |


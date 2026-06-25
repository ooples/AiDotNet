---
title: "ProgressiveGAN<T>"
description: "Production-ready Progressive GAN (ProGAN) implementation that generates high-resolution images by progressively growing the generator and discriminator during training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Production-ready Progressive GAN (ProGAN) implementation that generates high-resolution images
by progressively growing the generator and discriminator during training.

For Beginners:
Progressive GAN is a technique for training GANs that can generate very high-resolution
images (e.g., 1024x1024 pixels). Instead of trying to generate high-resolution images
from the start, it begins by generating small images (e.g., 4x4) and progressively
adds new layers to both the generator and discriminator to increase the resolution
(4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128 → 256x256 → 1024x1024).

Key innovations:

1. Progressive Growing: Start with low resolution and gradually add layers
2. Smooth Fade-in: New layers are faded in smoothly using a blending parameter (alpha)
3. Minibatch Standard Deviation: Helps prevent mode collapse by adding diversity
4. Equalized Learning Rate: Normalizes weights at runtime for better training dynamics
5. Pixel Normalization: Normalizes feature vectors in generator to prevent escalation

Based on "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
by Karras et al. (2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProgressiveGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,InputType,ILossFunction<>,Double,Double,ProgressiveGANOptions)` | Initializes a new instance of Progressive GAN. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the alpha value for smooth fade-in of new layers. |
| `CurrentResolutionLevel` | Gets the current resolution level (e.g., 0=4x4, 1=8x8, 2=16x16, etc.). |
| `Discriminator` | Gets the discriminator (critic) network that evaluates image quality. |
| `DiscriminatorLosses` | Gets the discriminator loss history. |
| `Generator` | Gets the generator network that produces images from latent codes. |
| `GeneratorLosses` | Gets the generator loss history. |
| `IsFadingIn` | Gets whether the network is currently in the fade-in phase after a growth step. |
| `LastGradientPenalty` | Gets the last computed gradient penalty value (for monitoring purposes only). |
| `LatentSize` | Gets the size of the latent vector (noise input). |
| `MaxResolutionLevel` | Gets the maximum resolution level the network can achieve. |
| `ParameterCount` | Gets the total number of trainable parameters in the ProgressiveGAN. |
| `UseMinibatchStdDev` | Gets or sets whether to use minibatch standard deviation. |
| `UsePixelNormalization` | Gets or sets whether to use pixel normalization in the generator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAlphaBlending(Tensor<>)` | Applies alpha blending between new and previous resolution outputs during fade-in. |
| `ApplyPixelNormalization(Tensor<>)` | Applies pixel normalization to feature maps using vectorized operations. |
| `ApplyVectorizedAdamUpdate(Vector<>,Vector<>,Vector<>,Vector<>,Int32,Double)` | Applies vectorized Adam update using Engine operations for SIMD/GPU acceleration. |
| `ClearLossHistory` | Clears the loss history. |
| `ComputeDriftPenalty(Tensor<>,Int32)` | Computes drift penalty to keep discriminator outputs near zero. |
| `ComputeGradientPenalty(Tensor<>,Tensor<>,Int32)` | Computes gradient penalty for Wasserstein GAN with gradient penalty. |
| `CreateDownsampledUpsampled(Tensor<>)` | Creates a downsampled then upsampled version of the output to simulate previous resolution. |
| `CreateFilledTensor(Int32[],)` | Creates a tensor filled with a constant value using vectorized Fill operation. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Generate(Int32)` | Generates images from random latent codes. |
| `Generate(Tensor<>)` | Generates images from specific latent codes. |
| `GenerateGaussianNoise(Int32)` | Generates Gaussian random noise for the generator input using Engine.GenerateGaussianNoise. |
| `GetCurrentResolution` | Gets the current image resolution based on the resolution level. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `GrowNetworks` | Grows the networks to the next resolution level. |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `ProjectLatentToGeneratorInputShape(Tensor<>)` | Projects a latent/noise tensor into the generator's DECLARED input volume (`Generator.Architecture.GetInputShape()`), copying the overlap and zero-padding the tail, with a leading batch axis. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTrainingMode(Boolean)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainStep(Tensor<>,Int32)` | Performs a single training step on a batch of real images. |
| `UpdateAlpha(Double)` | Updates the alpha value for progressive fade-in during training. |
| `UpdateDiscriminatorParametersVectorized` | Updates Discriminator parameters using vectorized Adam optimizer with Engine operations. |
| `UpdateGeneratorParametersVectorized` | Updates Generator parameters using vectorized Adam optimizer with Engine operations. |
| `UpdateParameters(Vector<>)` |  |


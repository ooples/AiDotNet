---
title: "WGANGP<T>"
description: "Represents a Wasserstein GAN with Gradient Penalty (WGAN-GP), an improved version of WGAN that uses gradient penalty instead of weight clipping to enforce the Lipschitz constraint."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Wasserstein GAN with Gradient Penalty (WGAN-GP), an improved version of WGAN
that uses gradient penalty instead of weight clipping to enforce the Lipschitz constraint.

## For Beginners

WGAN-GP is an enhanced version of WGAN with better training stability.

Key improvements over WGAN:

- Uses a "gradient penalty" instead of hard weight limits
- This penalty gently guides the critic to behave correctly
- More stable and reliable training
- Produces higher quality results
- Easier to use (fewer hyperparameters to tune)

The gradient penalty ensures the critic learns smoothly without the problems
that weight clipping can cause.

Reference: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)

## How It Works

WGAN-GP improves upon WGAN by:

- Replacing weight clipping with a gradient penalty term
- Providing smoother and more stable training
- Avoiding pathological behavior caused by weight clipping
- Achieving better performance and convergence
- Eliminating the need to tune the clipping threshold

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WGANGP(NeuralNetworkArchitecture<>,Double,Int32,WGANGPOptions)` | Creates a WGAN-GP with default generator and critic architectures derived from a single architecture. |
| `WGANGP(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,Int32,WGANGPOptions)` | Initializes a new instance of the `WGANGP` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Critic` | Gets the critic network that evaluates data quality. |
| `Generator` | Gets the generator network that creates synthetic data. |
| `ParameterCount` | Gets the total number of trainable parameters in the WGAN-GP. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradientPenaltyWithGradients(Tensor<>,Tensor<>,Int32)` | Computes the gradient penalty and returns both the penalty value and the parameter gradients. |
| `CreateNewInstance` |  |
| `CreateWGANGPArchitecture(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType)` | Creates the combined WGAN-GP architecture with correct dimension handling. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EvaluateModel(Int32)` | Evaluates the WGAN-GP by generating images and calculating metrics. |
| `GenerateImages(Tensor<>)` | Generates synthetic images using the generator. |
| `GenerateRandomNoiseTensor(Int32,Int32)` | Generates a tensor of random noise for the generator. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `ResetOptimizerState` | Resets both optimizer states for a fresh training run. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainCriticBatchWithGP(Tensor<>,Tensor<>,Int32)` | Trains the critic on a batch with gradient penalty. |
| `TrainStep(Tensor<>,Tensor<>)` | Performs one training step for the WGAN-GP using tensor batches. |
| `UpdateCriticWithOptimizer(Vector<>)` | Updates critic parameters using the configured optimizer with pre-computed gradients. |
| `UpdateGeneratorWithOptimizer` | Updates generator parameters using the configured optimizer. |
| `UpdateParameters(Vector<>)` | Updates the parameters of both the generator and critic networks. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_criticIterations` | The number of critic training iterations per generator iteration. |
| `_criticOptimizer` | The optimizer for the critic network. |
| `_generatorOptimizer` | The optimizer for the generator network. |
| `_gradientPenaltyCoefficient` | The coefficient for the gradient penalty term in the loss function. |


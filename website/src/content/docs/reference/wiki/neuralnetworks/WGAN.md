---
title: "WGAN<T>"
description: "Represents a Wasserstein Generative Adversarial Network (WGAN), which uses the Wasserstein distance (Earth Mover's distance) to measure the difference between the generated and real data distributions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Wasserstein Generative Adversarial Network (WGAN), which uses the Wasserstein distance
(Earth Mover's distance) to measure the difference between the generated and real data distributions.

## For Beginners

WGAN is an improved GAN that solves many training problems.

Key improvements over vanilla GAN:

- More stable training (less likely to fail)
- The loss value actually tells you how well training is going
- No mode collapse issues (generating only a few types of outputs)
- Can train the discriminator (critic) many times without problems

The main change is using a different mathematical way to measure the difference
between real and fake images, which turns out to be much more stable.

Reference: Arjovsky et al., "Wasserstein GAN" (2017)

## How It Works

WGAN addresses several training instabilities in vanilla GANs by:

- Using Wasserstein distance instead of Jensen-Shannon divergence
- Replacing the discriminator with a "critic" that doesn't output probabilities
- Enforcing a Lipschitz constraint through weight clipping
- Providing a loss that correlates with image quality
- Enabling more stable training and better convergence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,Int32,WGANOptions)` | Initializes a new instance of the `WGAN` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Critic` | Gets the critic network (called discriminator in vanilla GAN) that evaluates data. |
| `Generator` | Gets the generator network that creates synthetic data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClipCriticWeights` | Clips the critic's weights to enforce the Lipschitz constraint. |
| `CreateNetworkForInputType(NeuralNetworkArchitecture<>,InputType)` | Creates the appropriate neural network type based on the input type. |
| `CreateNewInstance` |  |
| `CreateWGANArchitecture(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType)` | Creates the combined WGAN architecture with correct dimension handling. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EvaluateModel(Int32)` | Evaluates the WGAN by generating images and calculating metrics. |
| `GenerateImages(Tensor<>)` | Generates synthetic images using the generator. |
| `GenerateRandomNoiseTensor(Int32,Int32)` | Generates a tensor of random noise for the generator. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `ResetOptimizerState` | Resets both optimizer states for a fresh training run. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainStep(Tensor<>,Tensor<>)` | Performs one training step for the WGAN using tensor batches. |
| `UpdateCriticWithOptimizer` | Updates critic parameters using the configured optimizer. |
| `UpdateGeneratorWithOptimizer` | Updates generator parameters using the configured optimizer. |
| `UpdateParameters(Vector<>)` | Updates the parameters of both the generator and critic networks. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_criticIterations` | The number of critic training iterations per generator iteration. |
| `_criticOptimizer` | The optimizer for the critic network. |
| `_generatorOptimizer` | The optimizer for the generator network. |
| `_weightClipValue` | The weight clipping threshold for enforcing the Lipschitz constraint. |


---
title: "InfoGAN<T>"
description: "Represents an Information Maximizing Generative Adversarial Network (InfoGAN), which learns disentangled representations in an unsupervised manner by maximizing mutual information between latent codes and generated observations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents an Information Maximizing Generative Adversarial Network (InfoGAN), which learns
disentangled representations in an unsupervised manner by maximizing mutual information
between latent codes and generated observations.

## For Beginners

InfoGAN learns to separate different features automatically.

Key concept:

- Splits random input into two parts:
1. Random noise (z): provides variety
2. Latent codes (c): control specific features
- Learns what each code controls WITHOUT labels
- Example: For faces, might learn codes for:
* Code 1: controls rotation
* Code 2: controls width
* Code 3: controls lighting

How it works:

- Generator uses both z and c to create images
- Auxiliary network Q tries to predict c from the generated image
- If Q can predict c accurately, the codes are meaningful
- This forces codes to represent interpretable features

Use cases:

- Discover semantic features in datasets
- Disentangled representation learning
- Controllable image generation
- Feature manipulation (change one aspect, keep others)

Reference: Chen et al., "InfoGAN: Interpretable Representation Learning by
Information Maximizing Generative Adversarial Nets" (2016)

## How It Works

InfoGAN extends the GAN framework by:

- Decomposing the input noise into incompressible noise (z) and latent codes (c)
- Maximizing the mutual information I(c; G(z,c)) between codes and generated images
- Learning interpretable and disentangled representations automatically
- Using an auxiliary network Q to approximate the posterior P(c|x)
- Enabling control over semantic features without labeled data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InfoGAN(NeuralNetworkArchitecture<>,Int32,Double,InfoGANOptions)` | Creates an InfoGAN with default architectures derived from a single architecture. |
| `InfoGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,InfoGANOptions)` | Initializes a new instance of the `InfoGAN` class with the specified architecture and training parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Discriminator` | Gets the discriminator network. |
| `Generator` | Gets the generator network. |
| `ParameterCount` | Gets the total number of trainable parameters in the InfoGAN. |
| `QNetwork` | Gets the auxiliary Q network that predicts latent codes from images. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendCodesIfNoiseOnly(Tensor<>)` | Appends a default-zero latent-code block to the input when the input is shaped as raw noise (last dim = generator.InputSize - latentCodeSize). |
| `CalculateBinaryGradients(Tensor<>,Tensor<>,Int32)` | Calculates gradients for binary cross-entropy. |
| `CalculateBinaryLoss(Tensor<>,Tensor<>,Int32)` | Calculates binary cross-entropy loss. |
| `CalculateMutualInfoGradients(Tensor<>,Tensor<>,Int32)` | Calculates gradients for mutual information loss. |
| `CalculateMutualInfoLoss(Tensor<>,Tensor<>,Int32)` | Calculates mutual information loss (MSE between predicted and true codes). |
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates noise and latent codes. |
| `ConfigureWeightLifetime(GpuOffloadOptions,IGpuOffloadAllocator)` | Forwards weight-lifetime configuration to the generator, discriminator, and Q-network sub-networks. |
| `CreateInfoGANArchitecture(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType)` | Creates the combined InfoGAN architecture with correct dimension handling. |
| `CreateLabelTensor(Int32,)` | Creates a label tensor. |
| `CreateNewInstance` | Creates a new instance of the InfoGAN with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes InfoGAN-specific data from a binary reader. |
| `ForwardForTraining(Tensor<>)` | Defines the InfoGAN forward graph for tape-based training. |
| `Generate(Tensor<>,Tensor<>)` | Generates images with specific latent codes. |
| `GenerateRandomLatentCodes(Int32)` | Generates random latent codes (continuous, uniform in [-1, 1]). |
| `GenerateRandomNoiseTensor(Int32,Int32)` | Generates random noise tensor using vectorized Gaussian noise generation with CPU/GPU acceleration. |
| `GetNamedLayerActivations(Tensor<>)` | Surfaces named activations from each sub-network so introspection invariants (NamedLayerActivations_ShouldBeNonEmpty) see real per-network state. |
| `GetOptions` |  |
| `GetParameterChunks` |  |
| `GetParameters` | Returns the concatenated trainable parameters from Generator, Discriminator, and QNetwork. |
| `ResetOptimizerState` | Resets the state of all optimizers to their initial values. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes InfoGAN-specific data to a binary writer. |
| `TrainStep(Tensor<>,Tensor<>,Tensor<>)` | Performs one training step for InfoGAN. |
| `UpdateDiscriminatorParameters` | Updates the parameters of the discriminator network using its optimizer. |
| `UpdateGeneratorParameters` | Updates the parameters of the generator network using its optimizer. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all networks in the InfoGAN. |
| `UpdateQNetworkParameters` | Updates the parameters of the Q network using its optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_discriminatorLosses` | List of recent discriminator losses for tracking training progress. |
| `_discriminatorOptimizer` | The optimizer used for training the discriminator network. |
| `_generatorLosses` | List of recent generator losses for tracking training progress. |
| `_generatorOptimizer` | The optimizer used for training the generator network. |
| `_latentCodeSize` | The size of the latent code c. |
| `_mutualInfoCoefficient` | The coefficient for the mutual information loss. |
| `_qNetworkOptimizer` | The optimizer used for training the Q network. |


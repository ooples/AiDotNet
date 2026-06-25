---
title: "ACGAN<T>"
description: "Represents an Auxiliary Classifier Generative Adversarial Network (AC-GAN), which extends conditional GANs by having the discriminator also predict the class label of the input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents an Auxiliary Classifier Generative Adversarial Network (AC-GAN), which extends
conditional GANs by having the discriminator also predict the class label of the input.

## For Beginners

AC-GAN generates specific types of images with better quality.

Key improvements over cGAN:

- Discriminator has two tasks: "Is it real?" AND "What class is it?"
- This dual task helps the discriminator learn better features
- Generator must create images that fool both checks
- Results in higher quality and more class-consistent images

Example use case:

- Generate digit "7" that looks very realistic
- Discriminator checks: 1) Is it real? 2) Is it a "7"?
- This forces the generator to make better "7"s

Reference: Odena et al., "Conditional Image Synthesis with Auxiliary Classifier GANs" (2017)

## How It Works

AC-GAN improves upon conditional GANs by:

- Making the discriminator predict both authenticity AND class label
- Providing stronger gradient signals for class-conditional generation
- Improving image quality and class separability
- Enabling better control over generated samples
- Training more stable than basic conditional GANs

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ACGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ACGANOptions)` | Initializes a new instance of the `ACGAN` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Discriminator` | Gets the discriminator network that predicts both authenticity and class. |
| `Generator` | Gets the generator network that creates class-conditional synthetic data. |
| `ParameterCount` | Gets the total number of trainable parameters in the ACGAN. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAuthenticityLoss(Tensor<>,Boolean,Int32)` | Calculates the authenticity loss (real vs fake). |
| `CalculateClassificationLoss(Tensor<>,Tensor<>,Int32)` | Calculates the classification loss for the class predictions using binary cross-entropy. |
| `CalculateDiscriminatorGradients(Tensor<>,Tensor<>,Boolean,Int32)` | Calculates gradients for discriminator backpropagation. |
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates noise and class labels for generator input. |
| `CreateOneHotLabels(Int32,Int32)` | Creates one-hot encoded class labels. |
| `CreateOneHotLabelsFromIndices(Int32,Int32[])` | Creates one-hot encoded label tensors from class indices. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes AC-GAN-specific data including networks and optimizer states. |
| `GenerateConditional(Tensor<>,Tensor<>)` | Generates class-conditional images. |
| `GenerateRandomNoiseTensor(Int32,Int32)` | Generates random noise tensor using vectorized Gaussian noise generation. |
| `GetOptions` |  |
| `NormalizeToProbabilities(Tensor<>)` | Normalizes discriminator output to valid probability range [0, 1]. |
| `ResetOptimizerState` | Resets both optimizer states for a fresh training run. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes AC-GAN-specific data including networks and optimizer states. |
| `Train(Tensor<>,Tensor<>)` | Performs a single training iteration using the standard neural network interface. |
| `TrainStep(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Performs one training step for the AC-GAN. |
| `UpdateDiscriminatorWithOptimizer` | Updates discriminator parameters using the configured optimizer. |
| `UpdateGeneratorWithOptimizer` | Updates generator parameters using the configured optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_discriminatorOptimizer` | The optimizer for the discriminator network. |
| `_generatorOptimizer` | The optimizer for the generator network. |
| `_numClasses` | The number of classes for classification. |


---
title: "DCGAN<T>"
description: "Represents a Deep Convolutional Generative Adversarial Network (DCGAN), an architecture that uses convolutional and transposed convolutional layers with specific design guidelines for stable training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Deep Convolutional Generative Adversarial Network (DCGAN), an architecture that uses
convolutional and transposed convolutional layers with specific design guidelines for stable training.

## For Beginners

DCGAN is an improved version of the basic GAN that uses specific
design patterns to make training more stable and produce higher quality images.

Key improvements over vanilla GAN:

- Uses convolutional layers specifically designed for images
- Includes batch normalization to stabilize training
- Follows proven architectural guidelines
- Produces sharper, more realistic images

Reference: Radford et al., "Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks" (2015)

## How It Works

DCGAN introduces several architectural constraints that improve training stability:

- Replace pooling layers with strided convolutions (discriminator) and fractional-strided

convolutions/transposed convolutions (generator)

- Use batch normalization in both generator and discriminator
- Remove fully connected hidden layers for deeper architectures
- Use ReLU activation in generator for all layers except output (uses Tanh)
- Use LeakyReLU activation in discriminator for all layers

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DCGAN(Int32,Int32,Int32,Int32,Int32,Int32,ILossFunction<>,DCGANOptions)` | Initializes a new instance of the `DCGAN` class with default DCGAN architecture. |
| `DCGAN(NeuralNetworkArchitecture<>,Int32,Int32,Int32,DCGANOptions)` | Creates a DCGAN with default dimensions derived from the architecture. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeInitialSpatialSize(Int32)` | Computes the initial spatial size for the generator based on target image size. |
| `CreateDCGANDiscriminatorArchitecture(Int32,Int32,Int32,Int32)` | Creates the architecture for the DCGAN discriminator following the original paper's guidelines. |
| `CreateDCGANGeneratorArchitecture(Int32,Int32,Int32,Int32,Int32)` | Creates the architecture for the DCGAN generator following the original paper's guidelines. |
| `CreateNewInstance` | Constructs a fresh DCGAN with the same paper-faithful hyperparameters so Clone / DeepCopy produces a deep-independent network whose layer list isn't shared with the original. |
| `GetOptions` |  |
| `IsPowerOfTwo(Int32)` | Checks if a number is a power of 2. |


---
title: "StyleGAN<T>"
description: "Represents a StyleGAN (Style-Based Generator Architecture for GANs) that generates high-quality images with fine-grained control over image style at different levels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a StyleGAN (Style-Based Generator Architecture for GANs) that generates
high-quality images with fine-grained control over image style at different levels.

## For Beginners

StyleGAN generates incredibly realistic images with fine control.

Key innovations:

- **Mapping Network**: Transforms random noise into style codes
- **Style Injection**: Injects style at each layer via AdaIN
- **Noise Injection**: Adds stochastic variation (hair, pores, etc.)
- **Style Mixing**: Combines styles from different sources
- **Progressive Growing**: Starts small, gradually adds detail

Architecture:

1. Mapping Network (Z → W): Transforms latent code to intermediate space
2. Synthesis Network: Generates image with style injection at each layer
3. Each layer: Upsample → Conv → AdaIN → Noise → Conv → AdaIN → Noise

Why it's better:

- Exceptional image quality
- Disentangled style control (separate coarse/fine features)
- Style mixing (combine different sources)
- Perceptual path length is shorter

Applications:

- High-quality face generation
- Style transfer and manipulation
- Image editing and synthesis
- Creative AI applications

Reference: Karras et al., "A Style-Based Generator Architecture for
Generative Adversarial Networks" (2019)

## How It Works

StyleGAN introduces several key innovations:

- Style-based generator with mapping network and synthesis network
- Adaptive Instance Normalization (AdaIN) for style injection
- Stochastic variation through noise injection
- Style mixing for disentangled control
- Progressive growing for high-resolution images
- State-of-the-art image quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StyleGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,Int32,InputType,ILossFunction<>,Double,Boolean,Double,StyleGANOptions)` | Initializes a new instance of the `StyleGAN` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Discriminator` | Gets the discriminator network. |
| `MappingNetwork` | Gets the mapping network that transforms Z to W. |
| `ParameterCount` | Gets the total number of trainable parameters in the StyleGAN. |
| `SynthesisNetwork` | Gets the synthesis network that generates images from styles. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateStyleGANArchitecture(Int32,NeuralNetworkArchitecture<>,InputType)` | Creates the combined StyleGAN architecture with correct dimension handling. |
| `Generate(Tensor<>)` | Generates images from latent codes. |
| `GenerateRandomLatentCodes(Int32)` | Generates random latent codes using vectorized Gaussian noise generation. |
| `GenerateWithStyleMixing(Tensor<>,Tensor<>)` | Generates images with style mixing. |
| `GetOptions` |  |
| `MixStyles(Tensor<>,Tensor<>)` | Mixes two sets of styles at a random layer. |
| `ReshapeForCNN(Tensor<>,NeuralNetworkArchitecture<>)` | Reshapes any-rank tensor to match CNN architecture input requirements. |
| `TrainStep(Tensor<>,Tensor<>)` | Performs one training step for StyleGAN. |
| `UpdateDiscriminatorParameters` | Updates Discriminator parameters using vectorized Adam optimizer. |
| `UpdateMappingNetworkParameters` | Updates MappingNetwork parameters using vectorized Adam optimizer. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all networks in the StyleGAN. |
| `UpdateSynthesisNetworkParameters` | Updates SynthesisNetwork parameters using vectorized Adam optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_enableStyleMixing` | Enables style mixing during training. |
| `_intermediateLatentSize` | The size of the intermediate latent code W. |
| `_latentSize` | The size of the latent code Z. |
| `_styleMixingProbability` | Probability of style mixing during training. |


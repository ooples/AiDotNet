---
title: "SAGAN<T>"
description: "Self-Attention GAN (SAGAN) implementation that uses self-attention mechanisms to model long-range dependencies in generated images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Self-Attention GAN (SAGAN) implementation that uses self-attention mechanisms
to model long-range dependencies in generated images.

For Beginners:
Traditional CNNs in GANs only look at nearby pixels (local receptive fields).
This works well for textures and local patterns, but struggles with global
structure and long-range relationships (like making sure both eyes of a face
look similar, or ensuring consistent geometric patterns).

Self-Attention solves this by letting each pixel "attend to" all other pixels,
similar to how Transformers work in NLP. Think of it as:

- CNN: "I can only see my immediate neighbors"
- Self-Attention: "I can see the entire image and decide what's important"

Example: When generating a dog's face:

- CNN: Might make one ear pointy and one floppy (inconsistent)
- SAGAN: Notices both ears and makes them match (consistent)

Key innovations:

1. Self-Attention Layers: Allow modeling of long-range dependencies
2. Spectral Normalization: Stabilizes training for both G and D
3. Hinge Loss: More stable than standard GAN loss
4. Two Time-Scale Update Rule (TTUR): Different learning rates for G and D
5. Conditional Batch Normalization: For class-conditional generation

Based on "Self-Attention Generative Adversarial Networks" by Zhang et al. (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAGAN(NeuralNetworkArchitecture<>,Int32,Int32,SAGANOptions)` | Creates a SAGAN with default architectures derived from a single architecture. |
| `SAGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32[],InputType,ILossFunction<>,Double,SAGANOptions)` | Initializes a new instance of Self-Attention GAN. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionLayers` | Gets the positions where self-attention layers are inserted. |
| `Discriminator` | Gets the discriminator network with self-attention layers. |
| `Generator` | Gets the generator network with self-attention layers. |
| `LatentSize` | Gets the size of the latent vector (noise input). |
| `NumClasses` | Gets the number of classes for conditional generation. |
| `ParameterCount` | Gets the total number of trainable parameters in the SAGAN. |
| `UseSpectralNormalization` | Gets or sets whether to use spectral normalization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAdamUpdate(ConvolutionalNeuralNetwork<>,Double,Boolean)` | Applies Adam optimizer update to network parameters using TTUR (Two Time-Scale Update Rule). |
| `CalculateHingeLoss(Tensor<>,Boolean,Int32)` | Calculates hinge loss for discriminator training. |
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates two tensors along the feature dimension. |
| `CreateClassEmbeddings(Int32[])` | Creates class embeddings for conditional generation. |
| `CreateNewInstance` |  |
| `CreateSAGANArchitecture(Int32,Int32,Int32,Int32,Int32,InputType)` | Creates the combined SAGAN architecture with correct dimension handling. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Generate(Int32,Int32[])` | Generates images from random latent codes. |
| `Generate(Tensor<>,Int32[])` | Generates images from specific latent codes. |
| `GenerateNoise(Int32)` | Generates random noise from a standard normal distribution. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTrainingMode(Boolean)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainStep(Tensor<>,Int32,Int32[])` | Performs a single training step on a batch of real images. |
| `UpdateParameters(Vector<>)` |  |


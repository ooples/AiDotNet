---
title: "BigGAN<T>"
description: "BigGAN implementation for large-scale high-fidelity image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

BigGAN implementation for large-scale high-fidelity image generation.

For Beginners:
BigGAN is a state-of-the-art GAN architecture that generates extremely high-quality
images by scaling up training in several ways:

1. Using very large batch sizes (256-2048 images at once)
2. Increasing model capacity (more parameters and feature maps)
3. Using class information to generate specific types of images

Think of it like training an artist:

- Small batch = showing the artist 1-2 examples at a time
- BigGAN batch = showing 256+ examples at once for better learning
- Class conditioning = telling the artist exactly what to draw ("draw a cat" vs "draw something")

Key innovations:

1. Large Batch Training: Uses batch sizes of 256-2048 (vs typical 32-128)
2. Spectral Normalization: Stabilizes training for both G and D
3. Self-Attention: Helps model long-range dependencies in images
4. Class Conditioning: Uses class embeddings for controlled generation
5. Truncation Trick: Trade diversity for quality at generation time
6. Orthogonal Initialization: Better weight initialization
7. Skip Connections: Direct paths in generator architecture

Based on "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
by Brock et al. (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BigGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,InputType,ILossFunction<>,Double,BigGANOptions)` | Initializes a new instance of BigGAN. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassEmbeddingDim` | Gets the dimension of class embeddings. |
| `Discriminator` | Gets the discriminator network that evaluates images and predicts their class. |
| `Generator` | Gets the generator network that produces images from noise and class labels. |
| `LatentSize` | Gets the size of the latent noise vector. |
| `NumClasses` | Gets the number of classes for conditional generation. |
| `ParameterCount` | Gets the total number of trainable parameters in the BigGAN. |
| `TruncationThreshold` | Gets or sets the truncation threshold for the truncation trick. |
| `UseSelfAttention` | Gets or sets whether to use self-attention layers. |
| `UseSpectralNormalization` | Gets or sets whether to use spectral normalization in both generator and discriminator. |
| `UseTruncation` | Gets or sets whether to use the truncation trick during generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyTruncation(Tensor<>)` | Applies the truncation trick to latent codes. |
| `CalculateHingeLoss(Tensor<>,Boolean,Int32)` | Calculates hinge loss for adversarial training. |
| `CalculateHingeLossGradients(Tensor<>,Boolean,Int32)` | Calculates hinge loss gradients for backpropagation. |
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates two tensors along the feature dimension. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiscriminatorPredictWithLabels(Tensor<>,Int32[])` | Gets the discriminator output with class conditioning using projection discriminator pattern. |
| `Generate(Int32)` | Generates random images with random class labels. |
| `Generate(Tensor<>,Int32[])` | Generates images from latent codes and class labels. |
| `GenerateGaussianNoise(Int32)` | Generates random noise for the generator input using vectorized Gaussian noise generation. |
| `GenerateWithGradients(Tensor<>,Int32[])` | Generates images for training with proper gradient tracking. |
| `GetClassEmbedding(Int32)` | Gets the class embedding for a specific class index. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeClassEmbeddings` | Initializes class embeddings using scaled uniform random initialization. |
| `InitializeDiscriminatorProjection` | Initializes the discriminator class projection matrix for the projection discriminator. |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `ProjectToGeneratorInputShape(Tensor<>)` | Projects a flat/concatenated (latent + class-embedding) tensor into the generator's DECLARED input volume (`Generator.Architecture.GetInputShape()`), copying the overlap and zero-padding the tail. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTrainingMode(Boolean)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainStep(Tensor<>,Int32[],Int32)` | Performs a single training step on a batch of real images with labels. |
| `UpdateDiscriminatorParameters` | Updates discriminator parameters using vectorized Adam optimizer. |
| `UpdateGeneratorParameters` | Updates generator parameters using vectorized Adam optimizer. |
| `UpdateParameters(Vector<>)` |  |


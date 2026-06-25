---
title: "ConditionalGAN<T>"
description: "Represents a Conditional Generative Adversarial Network (cGAN), which generates data conditioned on additional information such as class labels, attributes, or other contextual data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Conditional Generative Adversarial Network (cGAN), which generates data conditioned
on additional information such as class labels, attributes, or other contextual data.

## For Beginners

cGAN lets you control what kind of image is generated.

Key features:

- You can specify what you want to generate (e.g., "cat" vs. "dog")
- Both the generator and discriminator see the conditioning information
- Generator: "Given this label, create a matching image"
- Discriminator: "Is this image both real AND matching the label?"

Example use cases:

- Generate a specific digit (0-9) in MNIST
- Create images of specific object classes
- Generate faces with specific attributes (smiling, glasses, etc.)

Reference: Mirza and Osindero, "Conditional Generative Adversarial Nets" (2014)

## How It Works

Conditional GANs extend the basic GAN framework by:

- Conditioning both the generator and discriminator on additional information
- Allowing controlled generation (e.g., "generate a digit 7")
- Enabling class-conditional image synthesis
- Providing explicit control over the generated output characteristics

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConditionalGAN` | Initializes a new instance of the `ConditionalGAN` class with default configuration. |
| `ConditionalGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,ConditionalGANOptions)` | Initializes a new instance of the `ConditionalGAN` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateBinaryGradients(Tensor<>,Tensor<>)` | Calculates gradients for binary cross-entropy loss with logits. |
| `CalculateBinaryLoss(Tensor<>,Tensor<>)` | Calculates binary cross-entropy loss with logits (numerically stable). |
| `ConcatenateFlattenedImageAndCondition(Tensor<>,Tensor<>)` | Concatenates flattened images with condition vectors for 1D discriminator input. |
| `ConcatenateImageAndCondition(Tensor<>,Tensor<>)` | Concatenates images with condition vectors for discriminator input. |
| `ConcatenateSpatialImageAndCondition(Tensor<>,Tensor<>)` | Concatenates spatial images with condition vectors by tiling conditions across spatial dimensions and appending as extra channels. |
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates noise and condition vectors. |
| `CreateConditionalDiscriminatorArchitecture(NeuralNetworkArchitecture<>,Int32)` | Creates the discriminator architecture for conditional GAN. |
| `CreateConditionalGeneratorArchitecture(NeuralNetworkArchitecture<>,Int32)` | Creates the combined ConditionalGAN architecture with correct dimension handling. |
| `CreateLabelTensor(Int32,)` | Creates a label tensor filled with a specified value using vectorized fill. |
| `CreateNewInstance` |  |
| `CreateOneHotCondition(Int32,Int32)` | Creates a one-hot encoded condition tensor. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GenerateConditional(Tensor<>,Tensor<>)` | Generates images conditioned on specific labels. |
| `GenerateRandomNoiseTensor(Int32,Int32)` | Generates random noise tensor using vectorized Gaussian noise generation. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `PredictBatched(NeuralNetworkBase<>,Tensor<>)` | Runs Predict on a network, handling batched [B, N] input by processing per-sample. |
| `PredictCore(Tensor<>)` | Predicts output by treating the input as noise and adding default conditions. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the conditional GAN on a batch of data. |
| `TrainDiscriminatorOnBatch(Tensor<>,Tensor<>)` | Trains the discriminator on a batch of images. |
| `TrainGeneratorOnBatch(Tensor<>,Tensor<>,Tensor<>)` | Trains the generator on a batch. |
| `TrainStep(Tensor<>,Tensor<>,Tensor<>)` | Performs one training step for the conditional GAN. |
| `UpdateDiscriminatorWithOptimizer` | Updates discriminator parameters using the configured optimizer. |
| `UpdateGeneratorWithOptimizer` | Updates generator parameters using the configured optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numConditionClasses` | The number of condition classes/categories. |


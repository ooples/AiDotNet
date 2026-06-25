---
title: "Pix2Pix<T>"
description: "Represents a Pix2Pix GAN for paired image-to-image translation tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Pix2Pix GAN for paired image-to-image translation tasks.

## For Beginners

Pix2Pix transforms one type of image to another.

Key features:

- Learns from paired examples (input A becomes output B)
- Generator: U-Net architecture preserves spatial information
- Discriminator: PatchGAN focuses on local image patches
- Loss: Both "looks real" and "matches input"

Example use cases:

- Convert sketches to realistic photos
- Colorize black-and-white images
- Transform day scenes to night
- Semantic labels to photorealistic images
- Map to satellite image

Reference: Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks" (2017)

## How It Works

Pix2Pix is a conditional GAN for paired image-to-image translation:

- Uses a U-Net generator with skip connections
- Uses a PatchGAN discriminator that classifies image patches
- Combines adversarial loss with L1 reconstruction loss
- Requires paired training data (input-output pairs)
- Works for various tasks: edges to photo, day to night, sketch to image, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Pix2Pix(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,Pix2PixOptions)` | Initializes a new instance of the `Pix2Pix` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Discriminator` | Gets the PatchGAN discriminator network. |
| `Generator` | Gets the U-Net generator network. |
| `ParameterCount` | Gets the total number of trainable parameters in the Pix2Pix model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateBinaryGradients(Tensor<>,Tensor<>,Int32)` | Calculates the gradients for binary cross-entropy loss with logits. |
| `CalculateBinaryLoss(Tensor<>,Tensor<>,Int32)` | Calculates the binary cross-entropy loss with logits (numerically stable). |
| `CalculateL1Gradients(Tensor<>,Tensor<>)` | Calculates the gradients for L1 loss. |
| `CalculateL1Loss(Tensor<>,Tensor<>)` | Calculates the L1 loss between predictions and targets. |
| `ConcatenateFlattenedImages(Tensor<>,Tensor<>)` | Concatenates flattened image tensors along the feature dimension. |
| `ConcatenateImages(Tensor<>,Tensor<>)` | Concatenates two image tensors along the feature/channel dimension. |
| `ConcatenateSpatialImages(Tensor<>,Tensor<>)` | Concatenates spatial image tensors along the channel dimension. |
| `CreateLabelTensor(Int32,)` | Creates a tensor filled with a single label value. |
| `CreateNewInstance` |  |
| `CreatePix2PixArchitecture(NeuralNetworkArchitecture<>,InputType)` | Creates the combined Pix2Pix architecture with correct dimension handling. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `ResetOptimizerState` | Resets both optimizer states for a fresh training run. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainStep(Tensor<>,Tensor<>)` | Performs one training step for Pix2Pix. |
| `Translate(Tensor<>)` | Translates input images to output images. |
| `UpdateDiscriminatorWithOptimizer` | Updates discriminator parameters using the configured optimizer. |
| `UpdateGeneratorWithOptimizer` | Updates generator parameters using the configured optimizer. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all networks in the Pix2Pix GAN. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_discriminatorOptimizer` | The optimizer for the discriminator network. |
| `_generatorOptimizer` | The optimizer for the generator network. |
| `_l1Lambda` | The coefficient for the L1 reconstruction loss. |


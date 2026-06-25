---
title: "UNetDiscriminator<T>"
description: "U-Net Discriminator as used in Real-ESRGAN for improved perceptual quality."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

U-Net Discriminator as used in Real-ESRGAN for improved perceptual quality.

## For Beginners

The discriminator judges whether an image is real or fake.

Traditional discriminators output a single "real/fake" score for the whole image.
U-Net discriminator outputs a "real/fake" prediction for EVERY PIXEL, which:

- Provides more detailed feedback to the generator
- Helps produce sharper details and textures
- Enables better reconstruction of fine features

The U-Net architecture (encoder + decoder with skip connections) allows the
discriminator to consider both local details and global context.

## How It Works

This implements the U-Net discriminator from the Real-ESRGAN paper (Wang et al., 2021).
Unlike traditional patch discriminators, U-Net discriminator provides pixel-level feedback
which helps the generator produce finer details.

The architecture has an encoder-decoder structure:

**Reference:** Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UNetDiscriminator(Int32,Int32)` | Initializes a new U-Net discriminator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumBlocks` | Gets the number of encoder/decoder blocks. |
| `NumChannels` | Gets the base number of channels. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_convFirst` | Initial convolution. |
| `_convLast` | Final convolution (1x1 to output channels). |
| `_decoderBlocks` | Decoder blocks (upsampling path). |
| `_encoderBlocks` | Encoder blocks (downsampling path). |
| `_lastInput` | Cached input for backpropagation. |
| `_leakyReLU` | LeakyReLU activation. |
| `_numBlocks` | Number of encoder/decoder blocks. |
| `_numChannels` | Base number of channels. |
| `_skipConnections` | Skip connections stored during forward pass for concatenation. |


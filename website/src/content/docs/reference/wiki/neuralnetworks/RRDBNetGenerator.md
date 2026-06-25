---
title: "RRDBNetGenerator<T>"
description: "RRDBNet Generator - the full generator architecture from ESRGAN and Real-ESRGAN."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

RRDBNet Generator - the full generator architecture from ESRGAN and Real-ESRGAN.

## For Beginners

This is the "brain" of Real-ESRGAN that transforms low-resolution
images into high-resolution ones.

Key components:

- **Initial Conv**: Extracts basic features from the input image
- **RRDB Blocks**: 23 deep blocks that learn how to enhance details
- **Trunk Conv + Residual**: Combines deep features with initial features
- **Upsampling**: Makes the image bigger (2x or 4x depending on scale)
- **Final Convs**: Produces the final RGB output image

The default parameters (64 features, 32 growth, 23 RRDBs, 4x scale) are from the
Real-ESRGAN paper and produce excellent results for general image super-resolution.

## How It Works

This implements the complete RRDBNet generator from the ESRGAN paper (Wang et al., 2018).
It combines multiple RRDB blocks with upsampling for image super-resolution.

The architecture is:

**Reference:** Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RRDBNetGenerator(Int32,Int32,Int32,Int32,Int32,Int32,Double)` | Initializes a new RRDBNet generator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumFeatures` | Gets the number of feature channels. |
| `NumRRDBBlocks` | Gets the number of RRDB blocks. |
| `Scale` | Gets the upscaling factor. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `ApplyLeakyReLU(Tensor<>)` | Applies LeakyReLU activation. |
| `BackwardLeakyReLU(Tensor<>,Tensor<>)` | Backward pass through LeakyReLU. |
| `BuildConvNode(ConvolutionalLayer<>,ComputationNode<>,String)` | Builds a Conv2D computation node from a ConvolutionalLayer. |
| `Forward(Tensor<>)` |  |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_conv1Output` | Cached conv1 output for trunk residual. |
| `_convFirst` | Initial convolution: 3 → numFeatures. |
| `_convLast` | Final convolution: numFeatures → outputChannels (typically 3 for RGB). |
| `_hrConv` | High-resolution convolution (after upsampling). |
| `_inputChannels` | Number of input channels (3 for RGB). |
| `_lastInput` | Cached input for backpropagation. |
| `_leakyReLU` | LeakyReLU activation with negative slope 0.2. |
| `_numFeatures` | Number of feature channels. |
| `_outputChannels` | Number of output channels (3 for RGB). |
| `_pendingParameters` | Buffer for SetParameters when called pre-OnFirstForward. |
| `_pixelShuffleLayers` | PixelShuffle layers for upsampling. |
| `_rrdbBlocks` | The RRDB blocks for deep feature extraction. |
| `_rrdbOutputs` | Cached intermediate outputs for backpropagation. |
| `_scale` | Upscaling factor (2 or 4). |
| `_trunkConv` | Trunk convolution after RRDB blocks. |
| `_upsampleConvs` | Upsampling convolutions (one before each PixelShuffle). |
| `_upsampleOutputs` | Cached upsampling outputs for backpropagation. |


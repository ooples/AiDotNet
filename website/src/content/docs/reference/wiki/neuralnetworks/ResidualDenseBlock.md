---
title: "ResidualDenseBlock<T>"
description: "Residual Dense Block (RDB) as used in ESRGAN and Real-ESRGAN generators."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Residual Dense Block (RDB) as used in ESRGAN and Real-ESRGAN generators.

## For Beginners

This block is the building block of ESRGAN's generator.
It combines ideas from DenseNet (dense connections) and ResNet (residual learning):

1. **Dense connections**: Each conv layer can see ALL previous features
2. **Local residual**: The block's output is added to its input (helps training)
3. **Residual scaling**: The residual is scaled by 0.2 (prevents instability)
4. **LeakyReLU**: Uses LeakyReLU(0.2) instead of ReLU (better gradients)
5. **No batch norm**: ESRGAN generator doesn't use batch normalization

The default parameters (64 features, 32 growth, 0.2 scale) are from the paper.

## How It Works

This implements the Residual Dense Block from the ESRGAN paper (Wang et al., 2018).
It differs from DenseNet's Dense Block by using LeakyReLU, no batch normalization,
and a local residual connection with scaling.

The architecture consists of 5 convolutional layers with dense connections:

**Reference:** Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
ECCV 2018 Workshops. https://arxiv.org/abs/1809.00219

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResidualDenseBlock(Int32,Int32,Double)` | Initializes a new Residual Dense Block. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GrowthChannels` | Gets the growth channels (intermediate conv output channels). |
| `NumFeatures` | Gets the number of feature channels. |
| `ParameterCount` |  |
| `ResidualScale` | Gets the residual scaling factor. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddResidual(Tensor<>,Tensor<>,Double)` | Adds residual with scaling: output = a * scale + b. |
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `ApplyLeakyReLU(Tensor<>)` | Applies LeakyReLU activation to the input tensor. |
| `BackwardActivation(Tensor<>,Tensor<>)` | Backward pass through LeakyReLU activation. |
| `BuildConvNode(ConvolutionalLayer<>,ComputationNode<>,String)` | Builds a Conv2D computation node from a ConvolutionalLayer. |
| `ConcatenateChannels(Tensor<>,Tensor<>)` | Concatenates two tensors along the channel dimension. |
| `ConcatenateChannelsGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,IGpuBuffer,Int32,Int32,Int32,Int32)` | Concatenates two tensors along the channel dimension on GPU. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors. |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `ScaleGradient(Tensor<>,Double)` | Scales a tensor by a factor. |
| `ScaleNode(ComputationNode<>,Double,String)` | Scales a computation node by a scalar value using element-wise multiplication. |
| `SetParameters(Vector<>)` |  |
| `SplitGradient(Tensor<>,Int32,Int32)` | Splits a tensor along the channel dimension. |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activation` | LeakyReLU activation with negative slope 0.2 (from ESRGAN paper). |
| `_activationOutputs` | Cached intermediate activation outputs for backpropagation. |
| `_concatInputs` | Cached concatenated inputs to each conv layer for backpropagation. |
| `_convLayers` | The 5 convolutional layers in the dense block. |
| `_convOutputs` | Cached intermediate conv outputs (before activation) for backpropagation. |
| `_growthChannels` | Number of output channels for intermediate conv layers (growth rate). |
| `_inputHeight` | Input height — non-readonly: lazy ctor leaves -1 until OnFirstForward. |
| `_inputWidth` | Input width — non-readonly: lazy ctor leaves -1 until OnFirstForward. |
| `_lastInput` | Cached input for backpropagation. |
| `_numFeatures` | Number of input/output channels (feature channels). |
| `_residualScale` | Residual scaling factor. |


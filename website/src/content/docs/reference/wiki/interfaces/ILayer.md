---
title: "ILayer<T>"
description: "Defines the contract for neural network layers in the AiDotNet framework."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for neural network layers in the AiDotNet framework.

## How It Works

**For Beginners:** A neural network is made up of layers, similar to how a sandwich has layers.
Each layer processes data in some way and passes it to the next layer.

This interface defines what all layers must be able to do, regardless of their specific type.
Think of it as a checklist of abilities that every layer must have to work within our neural network.

## Properties

| Property | Summary |
|:-----|:--------|
| `CanExecuteOnGpu` | Gets whether this layer can execute on GPU. |
| `IsShapeResolved` | Indicates whether this layer's input/output shapes are concrete or still deferred. |
| `LayerName` | Gets the name of this layer for mixed-precision policy lookup. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsGpuTraining` | Gets whether this layer supports GPU-resident training (forward, backward, and parameter updates on GPU). |
| `SupportsTraining` | Indicates whether this layer supports training operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` | Clears all accumulated gradients in the layer. |
| `Deserialize(BinaryReader)` | Loads the layer's configuration and parameters from a binary stream. |
| `DownloadWeightsFromGpu` | Downloads the layer's weights and biases from GPU memory back to CPU. |
| `Forward(Tensor<>)` | Processes input data through the layer during the forward pass. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass, keeping the result on the GPU. |
| `ForwardWithPrecisionCheck(Tensor<>)` | Processes input data through the layer with automatic mixed-precision handling. |
| `GetActivationTypes` | Gets the activation functions used by this layer. |
| `GetBiases` | Gets the bias tensor for layers that have trainable biases. |
| `GetInputShape` | Gets the shape (dimensions) of the input data expected by this layer. |
| `GetOutputShape` | Gets the shape (dimensions) of the output data produced by this layer. |
| `GetParameterGradients` | Gets the gradients of all trainable parameters. |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetSubLayers` | Returns the immediate child layers contained within this layer. |
| `GetWeights` | Gets the weight tensor for layers that have trainable weights. |
| `ResetState` | Resets the internal state of the layer to its initial condition. |
| `Serialize(BinaryWriter)` | Saves the layer's configuration and parameters to a binary stream. |
| `SetParameters(Vector<>)` | Sets all trainable parameters of the layer to the specified values. |
| `SetTrainingMode(Boolean)` | Sets the layer to training or evaluation mode. |
| `UpdateParameters()` | Updates the layer's parameters using the specified learning rate. |
| `UpdateParameters(Vector<>)` | Updates the layer's parameters using the provided parameter values. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates the layer's parameters on GPU using the specified optimizer configuration. |
| `UploadWeightsToGpu` | Uploads the layer's weights and biases to GPU memory for GPU-resident training. |
| `ZeroGradientsGpu` | Resets the GPU gradient accumulators to zero. |


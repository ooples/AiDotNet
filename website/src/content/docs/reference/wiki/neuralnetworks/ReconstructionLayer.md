---
title: "ReconstructionLayer<T>"
description: "Represents a reconstruction layer that uses multiple fully connected layers to transform inputs into outputs."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a reconstruction layer that uses multiple fully connected layers to transform inputs into outputs.

## For Beginners

This layer works like a mini-network within your neural network.

Think of the ReconstructionLayer as a specialized team of artists:

- The first artist (first fully connected layer) makes a rough sketch
- The second artist (second fully connected layer) adds details to the sketch
- The third artist (third fully connected layer) finalizes the artwork

In an autoencoder network (a common use for this layer):

- Earlier layers compress your data into a compact form (like squeezing information)
- This reconstruction layer carefully expands that compact form back to the original size
- It learns how to restore information that was "squeezed out" during compression

For example, if you're building an image autoencoder, this layer would help transform
the compressed representation back into an image that looks like the original.

By using three layers instead of just one, this layer can learn more sophisticated
patterns for reconstructing the data.

## How It Works

The ReconstructionLayer is a composite layer that consists of three fully connected layers in sequence.
It is typically used in autoencoders or similar architectures to reconstruct the original input from a 
compressed representation. The layer provides a deeper transformation path through multiple hidden layers,
allowing it to learn more complex reconstruction functions than a single layer could.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReconstructionLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IActivationFunction<>)` | Initializes a new instance of the `ReconstructionLayer` class with scalar activation functions. |
| `ReconstructionLayer(Int32,Int32,Int32,Int32,IVectorActivationFunction<>,IVectorActivationFunction<>)` | Initializes a new instance of the `ReconstructionLayer` class with vector activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsShapeResolved` | ReconstructionLayer's own InputShape/OutputShape are concrete from construction, so the base `IsShapeResolved` returns `true` immediately — but its three lazy `FullyConnectedLayer` sub-layers have unallocated weight tensors until the first… |
| `ParameterCount` | Gets the total number of trainable parameters in the reconstruction layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(BinaryReader)` | Deserializes the reconstruction layer from a binary reader. |
| `FlattenToFeatureMatrix(Tensor<>)` | Normalises any incoming tensor to `[batch, featureWidth]` so the first `FullyConnectedLayer` always resolves to the same input width regardless of how the upstream layer shaped its output. |
| `Forward(Tensor<>)` | Performs the forward pass of the reconstruction layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass by chaining through sublayers. |
| `GetParameterGradients` | Sets the trainable parameters of the reconstruction layer. |
| `GetParameters` | Gets all trainable parameters of the reconstruction layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves the three lazy FCL sub-layers by routing a dummy tensor of `InputShape` through them. |
| `ResetState` | Resets the internal state of the reconstruction layer. |
| `Serialize(BinaryWriter)` | Serializes the reconstruction layer to a binary writer. |
| `UpdateParameters()` | Updates the parameters of the reconstruction layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_fc2` | The second fully connected layer in the reconstruction sequence. |
| `_fc3` | The third fully connected layer in the reconstruction sequence. |
| `_hidden1Dim` | The first fully connected layer in the reconstruction sequence. |
| `_useVectorActivation` | Flag indicating whether vector activation functions are used instead of scalar activation functions. |


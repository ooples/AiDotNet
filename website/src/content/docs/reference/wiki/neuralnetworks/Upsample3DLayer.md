---
title: "Upsample3DLayer<T>"
description: "Represents a 3D upsampling layer that increases the spatial dimensions of volumetric data using nearest-neighbor interpolation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a 3D upsampling layer that increases the spatial dimensions of volumetric data using nearest-neighbor interpolation.

## For Beginners

This layer makes 3D volumes larger by simply repeating voxel values.

Think of it like zooming in on a 3D image:

- When you zoom in on a voxelized object, each original voxel becomes a larger block
- This layer does the same thing to 3D feature volumes inside the neural network
- It's like stretching a 3D volume without adding any new information

For example, with a scale factor of 2:

- A 4×4×4 volume becomes an 8×8×8 volume
- Each voxel in the original volume is copied to a 2×2×2 block in the output
- This creates a larger volume that preserves the original content but with more voxels

This is essential for 3D U-Net decoder paths, where we need to progressively increase
the spatial resolution to match the original input size.

## How It Works

A 3D upsampling layer increases the spatial dimensions (depth, height, width) of volumetric tensors
by repeating values from the input to create a larger output. This implementation uses nearest-neighbor
interpolation, which copies each voxel value to fill a block in the output based on the scale factors.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Upsample3DLayer(Int32)` | Initializes a new instance of the `Upsample3DLayer` class with uniform scaling. |
| `Upsample3DLayer(Int32,Int32,Int32)` | Initializes a new instance of the `Upsample3DLayer` class with separate scale factors. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `ScaleDepth` | Gets the scale factor for the depth dimension. |
| `ScaleHeight` | Gets the scale factor for the height dimension. |
| `ScaleWidth` | Gets the scale factor for the width dimension. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32,Int32,Int32)` | Calculates the output shape based on input shape and scale factors. |
| `Clone` | Creates a deep copy of the layer with the same configuration. |
| `Deserialize(BinaryReader)` | Deserializes the layer from a binary stream. |
| `DeserializeFrom(BinaryReader)` | Creates a new Upsample3DLayer instance from serialized data. |
| `Forward(Tensor<>)` | Performs the forward pass of the 3D upsampling layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-resident forward pass of 3D upsampling, keeping all data on GPU. |
| `GetParameters` | Gets all trainable parameters. |
| `OnFirstForward(Tensor<>)` | Resolves channel/spatial dims and registers the resolved output shape on first forward. |
| `ResetState` | Resets the cached state from forward/backward passes. |
| `Serialize(BinaryWriter)` | Serializes the layer to a binary stream. |
| `SetParameters(Vector<>)` | Sets parameters from a vector. |
| `UpdateParameters()` | Updates parameters. |
| `ValidateParameters(Int32[],Int32,Int32,Int32)` | Validates constructor parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Whether batch dimension was added in ForwardGpu. |
| `_gpuInputShape` | Cached GPU input shape for backward pass. |
| `_lastInput` | The input tensor from the last forward pass, cached for backward computation. |


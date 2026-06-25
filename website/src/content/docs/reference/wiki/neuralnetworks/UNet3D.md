---
title: "UNet3D<T>"
description: "Represents a 3D U-Net neural network for volumetric semantic segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a 3D U-Net neural network for volumetric semantic segmentation.

## For Beginners

A 3D U-Net is like an intelligent 3D scanner that can identify and label
every single voxel (3D pixel) in a 3D volume.

Think of it like this:

- The encoder (left side of "U") looks at the big picture by progressively zooming out
- The decoder (right side of "U") zooms back in to produce detailed predictions
- Skip connections (horizontal lines in "U") preserve fine details from encoder to decoder

This is useful for:

- Medical imaging: Finding organs or tumors in CT/MRI scans
- 3D scene understanding: Segmenting objects in point clouds
- Part segmentation: Identifying different parts of 3D shapes

The "U" shape comes from the symmetric encoder-decoder design with skip connections.

## How It Works

A 3D U-Net extends the classic U-Net architecture to three dimensions for processing volumetric data.
It uses an encoder-decoder structure with skip connections to produce dense, per-voxel predictions
while preserving both local details and global context.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UNet3D` | Initializes a new instance with default architecture settings. |
| `UNet3D(NeuralNetworkArchitecture<>,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,UNet3DOptions)` | Initializes a new instance of the `UNet3D` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseFilters` | Gets the base number of filters in the first encoder block. |
| `NumClasses` | Gets the number of output classes (segmentation categories). |
| `NumEncoderBlocks` | Gets the number of encoder blocks in the network. |
| `VoxelResolution` | Gets the voxel grid resolution used by this network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this model type for cloning purposes. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary stream. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` | Gets metadata about this model for serialization and inspection. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the 3D U-Net. |
| `PredictEager(Tensor<>)` | Generates predictions for the given input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of input-output pairs. |
| `UpdateParameters(Vector<>)` | Updates the network parameters using a flat parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to compute the error between predictions and targets. |
| `_optimizer` | The optimizer used to update network parameters during training. |


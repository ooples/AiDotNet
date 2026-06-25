---
title: "VoxelCNN<T>"
description: "Represents a Voxel-based 3D Convolutional Neural Network for processing volumetric data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Voxel-based 3D Convolutional Neural Network for processing volumetric data.

## For Beginners

Think of a VoxelCNN as a 3D version of a regular image classifier.
Instead of looking at a 2D image, it examines a 3D grid of "blocks" (voxels) to understand
3D shapes. This is like how Minecraft represents the world - each block is either filled
or empty, and the pattern of blocks creates recognizable objects.

Applications include:

- Recognizing 3D objects from point cloud scans
- Detecting tumors in 3D medical scans
- Understanding room layouts from depth sensors

## How It Works

A Voxel CNN processes 3D volumetric data using 3D convolutions. This is useful for:

- 3D shape recognition from voxelized point clouds (e.g., ModelNet40)
- Medical image analysis (CT, MRI scans)
- Spatial occupancy prediction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoxelCNN` | Initializes a new instance of the `VoxelCNN` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseFilters` | Gets the base number of filters in the first convolutional layer. |
| `NumConvBlocks` | Gets the number of convolutional blocks in the network. |
| `VoxelResolution` | Gets the voxel grid resolution used by this network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this model type for cloning purposes. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary stream. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `ForwardForTraining(Tensor<>)` | Tape-recorded forward pass. |
| `GetModelMetadata` | Gets metadata about this model for serialization and inspection. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the VoxelCNN. |
| `PredictEager(Tensor<>)` | Generates predictions for the given input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Trains the network on a single batch of input-output pairs. |
| `UpdateParameters(Vector<>)` | Updates the network parameters using a flat parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lossFunction` | The loss function used to compute the error between predictions and targets. |
| `_optimizer` | The optimizer used to update network parameters during training. |


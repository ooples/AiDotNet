---
title: "SpiralNetOptions"
description: "Configuration options for SpiralNet++ mesh neural network."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for SpiralNet++ mesh neural network.

## For Beginners

These options control how the SpiralNet++ network
processes 3D mesh data. Key settings include:

- SpiralLength: How many neighbors to consider for each vertex
- ConvChannels: Feature sizes at each layer
- PoolRatios: How much to simplify the mesh at each pooling step

## How It Works

SpiralNet++ is a mesh convolution architecture that uses spiral sequences
to define consistent neighbor orderings on irregular mesh vertices.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvChannels` | Gets or sets the channel sizes for each convolution layer. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `FullyConnectedSizes` | Gets or sets the sizes of fully connected layers before output. |
| `IncludeCoordinates` | Gets or sets whether to include vertex coordinates as input features. |
| `IncludeNormals` | Gets or sets whether to include vertex normals as input features. |
| `InputFeatures` | Gets or sets the number of input features per vertex. |
| `NumClasses` | Gets or sets the number of output classes for classification. |
| `NumVertices` | Gets or sets the default vertex count used to synthesize fallback spiral indices when explicit mesh topology is not provided. |
| `PoolRatios` | Gets or sets the pooling ratios for mesh simplification. |
| `SpiralLength` | Gets or sets the length of the spiral sequence for convolutions. |
| `UseBatchNorm` | Gets or sets whether to use batch normalization. |
| `UseGlobalAveragePooling` | Gets or sets whether to use global average pooling before classification. |


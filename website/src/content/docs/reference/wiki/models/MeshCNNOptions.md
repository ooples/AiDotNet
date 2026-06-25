---
title: "MeshCNNOptions"
description: "Configuration options for the MeshCNN neural network."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the MeshCNN neural network.

## For Beginners

These options control how the MeshCNN network is configured.
The defaults are set to match the original paper and work well for most 3D shape
classification and segmentation tasks.

## How It Works

MeshCNN is a deep learning architecture for processing 3D mesh data. It operates
directly on the mesh structure using edge convolutions and mesh pooling operations.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvChannels` | Gets or sets the channel sizes for each edge convolution block. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `FullyConnectedSizes` | Gets or sets the sizes of fully connected layers before output. |
| `InputFeatures` | Gets or sets the number of input features per edge. |
| `LearningRate` | Gets or sets the initial learning rate for training. |
| `NumClasses` | Gets or sets the number of output classes for classification. |
| `NumNeighbors` | Gets or sets the number of neighboring edges to consider for each edge. |
| `PoolTargets` | Gets or sets the target edge counts after each pooling operation. |
| `UseBatchNorm` | Gets or sets whether to use batch normalization after each conv layer. |
| `UseGlobalAveragePooling` | Gets or sets whether to use global average pooling before FC layers. |


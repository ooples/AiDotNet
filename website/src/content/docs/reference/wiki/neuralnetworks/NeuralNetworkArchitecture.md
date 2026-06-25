---
title: "NeuralNetworkArchitecture<T>"
description: "Defines the structure and configuration of a neural network, including its layers, input/output dimensions, and task-specific properties."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Defines the structure and configuration of a neural network, including its layers, input/output dimensions, and task-specific properties.

## For Beginners

Think of NeuralNetworkArchitecture as the blueprint for building a neural network.

Just like an architect's blueprint for a building specifies:

- How many floors the building will have
- The size and purpose of each room
- How rooms connect to each other

The NeuralNetworkArchitecture defines:

- What kind of data your network will process (like images or text)
- How many layers your network will have
- How many neurons are in each layer
- How the layers connect to process your data

Before you can build a neural network, you need this blueprint to ensure all the parts
will fit together correctly. It helps prevent errors like trying to feed image data into
a network designed for text, or having layers that don't match up in size.

## How It Works

The NeuralNetworkArchitecture class serves as a blueprint for constructing neural networks with specific configurations.
It handles the validation of input dimensions, layer compatibility, and provides methods for retrieving information about
the network's structure. This architecture can be used to create various types of neural networks with different input 
dimensionalities and layer arrangements.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetworkArchitecture(InputType,NeuralNetworkTaskType,NetworkComplexity,Int32,Int32,Int32,Int32,Int32,List<ILayer<>>,Boolean,Int32,Int32,Int32)` | Initializes a new instance of the `NeuralNetworkArchitecture` class with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CalculatedInputSize` | Gets the calculated total size of the input based on dimensions. |
| `Complexity` | Gets the complexity level of the neural network. |
| `DefaultRandomSeedOverride` | Process-wide fallback for `RandomSeed` when no explicit per-architecture seed was set. |
| `HasDynamicSpatialDims` | Indicates whether this architecture declares dynamic (lazy) spatial dimensions — `InputHeight` and `InputWidth` set to `-1` as PyTorch-style sentinels, meaning the network resolves H/W from the actual input on the first forward. |
| `ImageEmbeddingDim` | Gets the dimensionality of image embeddings for multimodal networks. |
| `InputDepth` | Gets the depth dimension for 3D inputs. |
| `InputDimension` | Gets the dimensionality of the input (1, 2, or 3). |
| `InputFrames` | Gets the frame-count dimension for 4D (temporal video) inputs. |
| `InputHeight` | Gets the height dimension for 2D or 3D inputs. |
| `InputSize` | Gets or sets the size of the input vector. |
| `InputType` | Gets the type of input the neural network is designed to handle. |
| `InputWidth` | Gets the width dimension for 2D or 3D inputs. |
| `IsInitialized` | Gets a value indicating whether the architecture has been initialized. |
| `IsLayerOnly` | True when this architecture is a `CreateLayerOnly` stub — a placeholder used by sub-modules / lazy backbones that don't carry a semantic input contract. |
| `Layers` | Gets the optional list of predefined layers for the neural network. |
| `OutputSize` | Gets the size of the output vector. |
| `RandomSeed` | Optional seed for reproducible weight initialization across all layers in this architecture. |
| `ShouldReturnFullSequence` | Determines whether the network should return the full sequence or just the final output. |
| `TaskType` | Gets the type of task the neural network is designed to perform. |
| `TextEmbeddingDim` | Gets the dimensionality of text embeddings for multimodal networks. |
| `UseAutodiff` | Gets or sets a value indicating whether all layers in this architecture should use automatic differentiation by default. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputSize` | Calculates the total size of the output. |
| `CreateDynamicSpatial(InputType,NeuralNetworkTaskType,Int32,Int32,Int32)` | Creates an architecture for a network whose spatial dimensions are resolved on the first forward pass (PyTorch `LazyConv2d`-style). |
| `CreateLayerOnly` | Creates a "layer-only" architecture stub for sub-modules and detection backbones whose input contract is owned by a parent network. |
| `GetHiddenLayerSizes` | Gets the sizes of the hidden layers in the neural network. |
| `GetInputShape` | Gets the shape of the input as an array of dimensions. |
| `GetLayerSizes` | Gets the size of each layer in the neural network. |
| `GetOutputShape` | Gets the shape of the output as an array of dimensions. |
| `InitializeFromCachedData` | Initializes the architecture from cached data. |
| `IsEmbeddingCategoryLayer(ILayer<>)` | True when `layer` is an embedding-category layer (EmbeddingLayer, positional encoding, or a custom subclass). |
| `ValidateInputDimensions` | Validates the input dimensions to ensure they are consistent and appropriate for the selected input type. |


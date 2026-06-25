---
title: "GANDALFNetwork<T>"
description: "GANDALF (Gated Additive Neural Decision Forest) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

GANDALF (Gated Additive Neural Decision Forest) neural network for tabular data.

## For Beginners

GANDALF works like a smart forest of decision trees:

Architecture:

1. **Gating Network**: Learns which features are important (using FullyConnectedLayers)
2. **Neural Decision Trees**: Trees with learnable soft split decisions (SoftTreeLayers)
3. **Additive Ensemble**: Tree outputs are summed for final prediction

Key insight: Traditional trees have hard decisions (left or right).
GANDALF uses soft decisions where a sample partially goes both ways,
making the whole thing differentiable and trainable with gradient descent.

## How It Works

GANDALF combines gated feature selection with an ensemble of differentiable decision trees.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "GANDALF: Gated Adaptive Network for Deep Automated Learning of Features" (2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GANDALFNetwork` | Initializes a new GANDALF network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumTrees` | Gets the number of trees in the ensemble. |
| `Options` | Gets the GANDALF-specific options. |
| `TreeDepth` | Gets the tree depth. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetFeatureImportance` | Gets the learned feature importance from the soft decision trees. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the GANDALF network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the GANDALF network for the given input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the GANDALF network using the provided input and expected output. |
| `UpdateNetworkParameters` | Updates the parameters of all layers in the network based on computed gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |


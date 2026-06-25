---
title: "NODENetwork<T>"
description: "NODE (Neural Oblivious Decision Ensembles) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

NODE (Neural Oblivious Decision Ensembles) neural network for tabular data.

## For Beginners

NODE brings the interpretability of decision trees to deep learning:

Architecture:

1. **Feature Preprocessing**: Optional batch normalization and feature transformation
2. **Oblivious Trees**: Trees where all nodes at the same depth use the same feature
3. **Soft Splits**: Differentiable split decisions using entmax for sparse attention
4. **Ensemble Aggregation**: Multiple tree outputs combined for final prediction

Key insight: Traditional trees are hard to train with gradient descent.
NODE uses soft, differentiable splits that allow end-to-end training
while maintaining tree-like interpretability.

Example flow:
Features [batch, num_features] → Preprocessing [batch, hidden]
→ Tree 1 [batch, tree_out] \
→ Tree 2 [batch, tree_out] → Aggregate → Prediction
→ Tree N [batch, tree_out] /

## How It Works

NODE combines differentiable oblivious decision trees with neural network training.
Oblivious trees use the same splitting feature at all nodes of the same depth,
making them both interpretable and efficient.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data" (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NODENetwork` | Initializes a new NODE network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumLeaves` | Gets the number of leaf nodes per tree. |
| `NumTrees` | Gets the number of trees in the ensemble. |
| `Options` | Gets the NODE-specific options. |
| `TreeDepth` | Gets the tree depth. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |


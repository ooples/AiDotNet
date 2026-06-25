---
title: "TabMNetwork<T>"
description: "TabM (Parameter-Efficient Ensemble) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabM (Parameter-Efficient Ensemble) neural network for tabular data.

## For Beginners

TabM gets ensemble benefits efficiently:

Architecture:

1. **Shared Base Weights**: Core network weights shared by all ensemble members
2. **Rank Vectors**: Small per-member vectors that customize each member
3. **Weight Modulation**: Rank vectors scale the shared weights
4. **Ensemble Aggregation**: Average or concatenate member predictions

Key insight: Traditional ensembles need N times the parameters for N models.
TabM only adds about 1-5% extra parameters per member while getting similar
benefits. This is like having multiple experts share most of their knowledge
but each having their own "style" or perspective.

## How It Works

TabM uses BatchEnsemble-style parameter sharing to create multiple ensemble members
with minimal parameter overhead. Each member shares base weights but has its own
small rank vectors that modulate the shared weights.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabMNetwork` | Initializes a new TabM network with default configuration. |
| `TabMNetwork(NeuralNetworkArchitecture<>,TabMOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double)` | Initializes a new TabM network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumEnsembleMembers` | Gets the number of ensemble members. |
| `Options` | Gets the TabM-specific options. |

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


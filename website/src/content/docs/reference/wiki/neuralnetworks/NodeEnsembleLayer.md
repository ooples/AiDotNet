---
title: "NodeEnsembleLayer<T>"
description: "NODE ensemble: a set of differentiable oblivious decision trees run in PARALLEL on the same input, with their outputs concatenated (Popov et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

NODE ensemble: a set of differentiable oblivious decision trees run in PARALLEL on the same
input, with their outputs concatenated (Popov et al. 2019, "Neural Oblivious Decision Ensembles
for Deep Learning on Tabular Data").

## How It Works

Each of the `numTrees` trees is an `ObliviousDecisionTreeLayer` that sees the
full feature vector and produces a `[batch, treeOutputDim]` output; the ensemble
concatenates them into `[batch, numTrees * treeOutputDim]`, which a linear head maps to the
prediction. This is the defining NODE structure — a parallel ensemble — as opposed to stacking
trees sequentially (which would feed one tree's output into the next, a dimension mismatch and
not an ensemble).

Implemented as one composite layer: the trees are registered via
`ILayer{` and the forward is tape-recorded Engine ops, so
gradients reach every tree. Feature count adapts to the fed width via a rebuild.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NodeEnsembleLayer(Int32,Int32,Int32,Int32,Double)` | Initializes a NODE ensemble. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |


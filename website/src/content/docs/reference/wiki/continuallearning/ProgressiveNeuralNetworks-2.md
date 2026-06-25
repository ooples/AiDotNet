---
title: "ProgressiveNeuralNetworks<T, TInput, TOutput>"
description: "Progressive Neural Networks (PNN) strategy for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

Progressive Neural Networks (PNN) strategy for continual learning.

## For Beginners

PNN prevents catastrophic forgetting by creating
a new "column" (copy of the network) for each new task. Previous columns
are frozen, so old knowledge is never overwritten. New columns can use
lateral connections to leverage features from previous columns.

## How It Works

**How it works:**

**The Math:**

For layer l in column k:

h_k^l = f(W_k^l * h_k^(l-1) + Σ_{j<k} U_{k,j}^l * h_j^(l-1))

Where U_{k,j}^l are the lateral connection weights from column j to column k

**Comparison to Other Methods:**

PNN guarantees no forgetting but has linear memory growth with tasks.
Best for scenarios with a small number of tasks where zero forgetting is critical.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProgressiveNeuralNetworks(ILossFunction<>)` | Initializes a new PNN strategy with default options. |
| `ProgressiveNeuralNetworks(ILossFunction<>,PNNOptions<>)` | Initializes a new PNN strategy with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnCount` | Gets the number of columns (one per task). |
| `LateralScaling` | Gets the lateral connection scaling factor. |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `RequiresMemoryBuffer` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` |  |
| `ComputeLateralConnectionSize` | Computes the size of lateral connections needed. |
| `ComputeLateralContribution(List<Vector<>>,Int32)` | Computes the forward pass through lateral connections. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetColumnInfo(Int32)` | Gets information about a specific column. |
| `GetStateForSerialization` |  |
| `GetTotalParameterCount` | Gets the total parameter count across all columns. |
| `InitializeLateralWeights(Int32)` | Initializes lateral connection weights with small random values. |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |


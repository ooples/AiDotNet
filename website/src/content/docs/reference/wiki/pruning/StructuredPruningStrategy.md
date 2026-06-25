---
title: "StructuredPruningStrategy<T>"
description: "Structured pruning removes entire neurons/filters/channels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Pruning`

Structured pruning removes entire neurons/filters/channels.

## For Beginners

This strategy removes entire building blocks, not just individual connections.

The difference between structured and unstructured pruning:

Unstructured pruning (like magnitude or gradient):

- Removes individual connections randomly scattered throughout the network
- Creates a "swiss cheese" pattern with holes everywhere
- Requires special sparse matrix libraries to run faster
- Harder to deploy on mobile or edge devices

Structured pruning:

- Removes entire neurons, filters, or channels
- Creates a smaller but still dense (solid) network
- Runs faster on ANY hardware - no special libraries needed!
- Easier to deploy and understand

Analogy: Building a smaller car

- Unstructured: Remove random bolts and parts everywhere (car still same size, just hollow)
- Structured: Remove entire seats or components (car is actually smaller)

Types of structured pruning:

1. **Neuron pruning**: Remove entire neurons (columns in weight matrix)
- Reduces layer width
- Common in fully connected layers

2. **Filter pruning**: Remove entire convolutional filters
- Reduces number of feature maps
- Very effective for CNNs

3. **Channel pruning**: Remove input/output channels
- Reduces both computation and memory
- Commonly used with filter pruning

Example:
Original layer: 100 neurons
After 40% structured pruning: 60 neurons (actually smaller!)
After 40% unstructured pruning: 100 neurons (60% of weights are zero, but layer size unchanged)

Trade-offs:

- Structured pruning: Less flexibility, but real speedups
- Unstructured pruning: More flexibility, but needs special hardware/software

## How It Works

Structured pruning removes entire structural units (neurons, filters, channels) rather than
individual weights. This results in smaller dense networks that are easier to deploy and
can achieve actual speedups on standard hardware, unlike unstructured pruning which creates
sparse matrices that require specialized libraries for acceleration.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StructuredPruningStrategy(StructuredPruningStrategy<>.StructurePruningType)` | Creates a new structured pruning strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsStructured` | Gets whether this is structured pruning (true). |
| `Name` | Gets the name of this pruning strategy. |
| `RequiresGradients` | Gets whether this strategy requires gradients (false for structured pruning). |
| `SupportedPatterns` | Gets supported sparsity patterns for structured pruning. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyLayerAwarePruning(IFullModel<,Tensor<>,Tensor<>>,PruningConfig)` | Applies layer-aware structured pruning to a model using per-category sparsity targets. |
| `ApplyPruning(Matrix<>,IPruningMask<>)` | Applies the pruning mask to weights in-place. |
| `ApplyPruning(Tensor<>,IPruningMask<>)` | Applies pruning mask to tensor weights in-place. |
| `ApplyPruning(Vector<>,IPruningMask<>)` | Applies pruning mask to vector weights in-place. |
| `ComputeImportanceScores(Matrix<>,Matrix<>)` | Computes importance scores for structural units. |
| `ComputeImportanceScores(Tensor<>,Tensor<>)` | Computes importance scores for tensor weights. |
| `ComputeImportanceScores(Vector<>,Vector<>)` | Computes importance scores for vector weights. |
| `Create2to4Mask(Tensor<>)` | Creates a 2:4 structured sparsity mask (NVIDIA Ampere compatible). |
| `CreateMask(Matrix<>,Double)` | Creates a structured pruning mask. |
| `CreateMask(Tensor<>,Double)` | Creates a pruning mask for tensor weights. |
| `CreateMask(Vector<>,Double)` | Creates a pruning mask for vector weights. |
| `CreateNtoMMask(Tensor<>,Int32,Int32)` | Creates an N:M structured sparsity mask. |
| `ToSparseFormat(Tensor<>,SparseFormat)` | Converts pruned weights to sparse format for efficient storage. |
| `ToSparseFormat(Tensor<>,SparseFormat,Int32,Int32)` | Converts pruned weights to N:M structured sparse format for efficient storage. |


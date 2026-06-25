---
title: "TapeTrainingStep<T>"
description: "Provides PyTorch-style training step using tape-based automatic differentiation, with two-level caching for parameter collection that outperforms PyTorch's `model.parameters()` which rebuilds the full list on every call."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Training`

Provides PyTorch-style training step using tape-based automatic differentiation,
with two-level caching for parameter collection that outperforms PyTorch's
`model.parameters()` which rebuilds the full list on every call.

## How It Works

**Performance advantages over PyTorch:**

Both caches are invalidated via a version counter that increments when
the layer structure changes (layers added/removed).

## Methods

| Method | Summary |
|:-----|:--------|
| `CollectParameters(IEnumerable<ILayer<>>,Int32)` | Collects all trainable parameters by recursively walking layers and their sub-layers. |
| `CollectRecursive(IEnumerable<ILayer<>>,List<Tensor<>>,HashSet<Tensor<>>)` | Recursively walks layers and sub-layers to collect all ITrainableLayer parameters. |
| `CollectTrainableLayers(IEnumerable<ILayer<>>,Int32)` | Collects all ITrainableLayer instances for ZeroGrad. |
| `ComputeTopologyFingerprint(IList<ILayer<>>)` | Computes a topology fingerprint over the layer list using FNV-1a 64-bit. |
| `InvalidateCache` | Invalidates all caches. |
| `Step(IReadOnlyList<ITrainableLayer<>>,Tensor<>,Tensor<>,,Func<Tensor<>,Tensor<>>,Func<Tensor<>,Tensor<>,Tensor<>>)` | Executes a single training step using tape-based autodiff. |
| `ZeroGradAll(IEnumerable<ILayer<>>,Int32)` | Zeros gradients for all trainable layers, including those nested in composite layers. |


---
title: "GradientCompressionOptimizer<T, TInput, TOutput>"
description: "GradientCompressionOptimizer<T, TInput, TOutput> — Models & Types in AiDotNet.DistributedTraining."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientCompressionOptimizer(IOptimizer<,,>,IShardingConfiguration<>,Double,Boolean,Boolean)` | Creates a gradient compression optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyQuantization(Vector<>)` | Applies quantization compression. |
| `ApplyTopKSparsification(Vector<>)` | Applies top-k sparsification - keeps only largest magnitude values. |
| `CompressGradients(Vector<>)` | Compresses gradients using configured compression techniques. |
| `DecompressGradients(Vector<>,Int32)` | Decompresses gradients back to full format. |
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |


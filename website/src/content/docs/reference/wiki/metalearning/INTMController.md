---
title: "INTMController<T>"
description: "Interface for NTM controller."
section: "API Reference"
---

`Interfaces` · `AiDotNet.MetaLearning.Algorithms`

Interface for NTM controller.

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>,List<Tensor<>>)` | Forward pass through the controller. |
| `GenerateAddVector(Tensor<>)` | Generates add vector for writing. |
| `GenerateEraseVector(Tensor<>)` | Generates erase vector for writing. |
| `GenerateOutput(Tensor<>,List<Tensor<>>)` | Generates final output. |
| `GenerateReadKeys(Tensor<>)` | Generates read keys for all read heads. |
| `GenerateWriteKey(Tensor<>)` | Generates write key. |
| `GetParameters` | Gets controller parameters. |
| `Reset` | Resets controller state. |
| `SetParameters(Vector<>)` | Sets controller parameters (updates internal weights). |


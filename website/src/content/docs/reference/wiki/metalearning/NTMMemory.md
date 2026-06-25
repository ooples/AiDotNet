---
title: "NTMMemory<T>"
description: "External memory matrix for Neural Turing Machine."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

External memory matrix for Neural Turing Machine.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NTMMemory(Int32,Int32,Nullable<Int32>)` | Initializes a new instance of NTMMemory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Size` | Gets the size (number of slots) of the memory. |
| `Width` | Gets the width (dimension) of each memory slot. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Clones the memory matrix. |
| `ComputeSharpnessPenalty` | Computes memory sharpness penalty for regularization. |
| `ComputeUsagePenalty` | Computes memory usage penalty for regularization. |
| `GetValue(Int32,Int32)` | Gets the memory value at the specified location. |
| `InitializeLearned` | Initializes memory with learned initialization. |
| `InitializeRandom` | Initializes memory with random values. |
| `InitializeZeros` | Initializes memory with zeros. |
| `Read(Vector<>)` | Reads from memory using attention weights. |
| `Reset` | Resets memory to initial state. |
| `Write(Vector<>,Tensor<>,Tensor<>)` | Writes to memory using interpolation. |


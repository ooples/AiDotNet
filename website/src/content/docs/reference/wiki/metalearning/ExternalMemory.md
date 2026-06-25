---
title: "ExternalMemory<T>"
description: "External memory matrix for MANN."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

External memory matrix for MANN.

## How It Works

Stores key-value pairs that can be read and written by the MANN controller.
Uses attention-based reading and LRU-based writing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExternalMemory(Int32,Int32,Int32,INumericOperations<>)` | Initializes a new instance of the ExternalMemory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KeySize` | Gets the key dimension. |
| `Size` | Gets the memory size. |
| `ValueSize` | Gets the value dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearRecent(Double)` | Clears recent memories based on retention ratio. |
| `ClearSlot(Int32)` | Clears a specific memory slot. |
| `Clone` | Clones the external memory. |
| `ComputeUsagePenalty` | Computes memory usage penalty for regularization. |
| `FindLeastRecentlyUsedSlot` | Finds the least recently used memory slot. |
| `FindRarelyUsedSlots(Double)` | Finds rarely used memory slots. |
| `GetKey(Int32)` | Gets memory key at specified index. |
| `Read(Vector<>)` | Reads from memory using attention weights. |
| `Write(Int32,Vector<>,Vector<>)` | Writes to memory at specified location. |


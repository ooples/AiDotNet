---
title: "IMemoryBasedStrategy<T, TInput, TOutput>"
description: "Extended strategy interface for strategies that store task examples."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ContinualLearning.Interfaces`

Extended strategy interface for strategies that store task examples.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxExamples` | Gets the maximum number of examples that can be stored. |
| `StoredExampleCount` | Gets the number of examples stored in memory. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearMemory` | Clears all stored examples. |
| `SampleExamples(Int32)` | Samples a batch of stored examples for replay. |
| `StoreTaskExamples(IDataset<,,>)` | Stores examples from a completed task. |


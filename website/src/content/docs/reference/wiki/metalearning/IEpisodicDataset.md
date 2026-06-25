---
title: "IEpisodicDataset<T, TInput, TOutput>"
description: "IEpisodicDataset<T, TInput, TOutput> — Interfaces in AiDotNet.MetaLearning.Data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.MetaLearning.Data`

_No summary documentation available yet._

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassCounts` | Gets the number of examples per class in the dataset. |
| `NumClasses` | Gets the total number of classes available in the dataset. |
| `Split` | Gets the split type of this dataset (train, validation, or test). |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleTasks(Int32,Int32,Int32,Int32)` | Samples a batch of N-way K-shot tasks from the dataset. |
| `SetRandomSeed(Int32)` | Sets the random seed for reproducible task sampling. |


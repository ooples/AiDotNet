---
title: "DatasetMixer<T>"
description: "Blends multiple datasets by weight ratios, producing mixed batches for curriculum or domain mixing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Multimodal`

Blends multiple datasets by weight ratios, producing mixed batches for curriculum or domain mixing.

## For Beginners

Imagine you have three types of training data: books, code, and
conversations. You want your model to see 50% books, 30% code, and 20% conversations.
DatasetMixer handles this:

## How It Works

Dataset mixing is a crucial technique for training large language models and multimodal models.
It allows combining data from multiple sources (domains) with controlled ratios, enabling:

- Domain balancing (upweight rare domains)
- Curriculum learning (gradually shift domain mix during training)
- Multi-task training (combine datasets for different tasks)

## Properties

| Property | Summary |
|:-----|:--------|
| `NormalizedWeights` | Gets the normalized mixing weights (summing to 1.0). |
| `SourceCount` | Gets the number of data sources in the mixer. |
| `TotalSamples` | Gets the total number of samples across all sources. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSource(MultimodalDataset<>,Double,String)` | Adds a dataset source with a mixing weight. |
| `AddTensorSource(Tensor<>,Tensor<>,ModalityType,Double,String)` | Adds a tensor-based data source with a mixing weight. |
| `GetMixedBatchIndices(Int32,Nullable<Int32>)` | Generates batch indices following the mixing weights distribution. |
| `GetMixedBatches(Int32,Nullable<Int32>)` | Generates an infinite stream of mixed batch indices, following mixing weights. |
| `GetSample(Int32,Int32)` | Gets a sample from a specific source at a specific index. |
| `GetSourceInfo` | Gets information about all sources. |
| `UpdateWeight(Int32,Double)` | Updates the mixing weight for a specific source (for curriculum learning). |


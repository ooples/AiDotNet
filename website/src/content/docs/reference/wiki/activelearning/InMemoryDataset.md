---
title: "InMemoryDataset<T, TInput, TOutput>"
description: "In-memory implementation of a dataset for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Data`

In-memory implementation of a dataset for active learning.

## For Beginners

This is the most common dataset implementation.
It stores all data in memory, making it fast to access but limited by
available RAM. Suitable for most datasets that fit in memory.

## How It Works

**Features:**

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InMemoryDataset([],[])` | Creates a new labeled dataset. |
| `InMemoryDataset([],[],Boolean)` | Creates a new dataset with optional labeling. |
| `InMemoryDataset([],[],Boolean,[],String[],[],DatasetMetadata,Int32,Boolean)` | Creates a new dataset with full configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassCount` |  |
| `ClassLabels` |  |
| `Count` |  |
| `FeatureCount` |  |
| `FeatureNames` |  |
| `HasLabels` |  |
| `Inputs` |  |
| `IsClassification` |  |
| `Metadata` |  |
| `Outputs` |  |
| `SampleWeights` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSamples([],[])` |  |
| `Clone` |  |
| `CreateClassification([],[],[])` | Creates a classification dataset with class labels. |
| `CreateEmpty` | Creates an empty dataset. |
| `CreateUnlabeled([])` | Creates an unlabeled dataset. |
| `CreateWeighted([],[],[])` | Creates a dataset with sample weights. |
| `CreateWithMetadata([],[],DatasetMetadata)` | Creates a dataset with metadata. |
| `Except(Int32[])` |  |
| `GetIndices` |  |
| `GetInput(Int32)` |  |
| `GetOutput(Int32)` |  |
| `GetSample(Int32)` |  |
| `Merge(IDataset<,,>)` |  |
| `RemoveSamples(Int32[])` |  |
| `Shuffle(Random)` |  |
| `Split(Double,Random)` |  |
| `Subset(Int32[])` |  |
| `UpdateLabels(Int32[],[])` |  |


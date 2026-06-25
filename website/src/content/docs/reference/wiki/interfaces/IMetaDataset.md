---
title: "IMetaDataset<T, TInput, TOutput>"
description: "Represents a high-level meta-dataset that can generate episodes for meta-learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a high-level meta-dataset that can generate episodes for meta-learning.
Unlike `IEpisodicDataset` which operates on pre-built episodes,
this interface generates episodes on-the-fly from an underlying data source.

## For Beginners

A meta-dataset is a collection of data organized so that we can create
many different learning tasks from it. Each task (episode) contains a small training set (support)
and a small test set (query). The meta-learner trains across many such tasks to learn how to
learn quickly from small amounts of data.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassExampleCounts` | Gets the number of examples available for each class, keyed by class index. |
| `Name` | Gets the name of this meta-dataset. |
| `TotalClasses` | Gets the total number of distinct classes available in the dataset. |
| `TotalExamples` | Gets the total number of examples across all classes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleEpisode(Int32,Int32,Int32)` | Samples a single episode from the dataset. |
| `SampleEpisodes(Int32,Int32,Int32,Int32)` | Samples multiple episodes from the dataset. |
| `SetSeed(Int32)` | Sets the random seed for reproducible episode generation. |
| `SupportsConfiguration(Int32,Int32,Int32)` | Gets whether this dataset supports a given N-way K-shot configuration. |


---
title: "IEpisodicDataset<T, TInput, TOutput>"
description: "Interface for episodic datasets used in meta-learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for episodic datasets used in meta-learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassesPerEpisode` | Gets the number of classes per episode (N-way). |
| `EpisodeCount` | Gets the total number of episodes in the dataset. |
| `ExamplesPerClass` | Gets the number of examples per class (K-shot). |
| `HasMoreEpisodes` | Gets whether the dataset has more episodes. |
| `QueryExamplesPerClass` | Gets the number of query examples per class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetEpisode(Int32)` | Gets a specific episode from the dataset. |
| `GetEpisodeBatch(Int32,Boolean)` | Gets a batch of episodes from the dataset. |
| `Reset` | Resets the dataset to the beginning. |
| `SampleTasks(Int32,Nullable<Int32>)` | Samples a batch of tasks from the dataset. |
| `SetRandomSeed(Int32)` | Sets the random seed for reproducible sampling. |
| `Shuffle` | Shuffles the dataset. |


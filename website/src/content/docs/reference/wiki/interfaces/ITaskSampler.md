---
title: "ITaskSampler<T, TInput, TOutput>"
description: "Controls the strategy for sampling meta-learning tasks (episodes) from a dataset."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Controls the strategy for sampling meta-learning tasks (episodes) from a dataset.
Implementations can provide uniform, balanced, curriculum-based, or dynamic sampling.

## For Beginners

A task sampler decides *which* tasks (episodes) to give
the meta-learner during training. Different strategies can make training faster or more robust:

- **Uniform**: Picks tasks completely at random — simple and effective.
- **Balanced**: Ensures every class appears equally often across tasks.
- **Dynamic**: Focuses on tasks the model struggles with the most.
- **Curriculum**: Starts with easy tasks, gradually increases difficulty.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumQueryPerClass` | Gets the number of query examples per class used by this sampler. |
| `NumShots` | Gets the K-shot configuration used by this sampler. |
| `NumWays` | Gets the N-way configuration used by this sampler. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleBatch(Int32)` | Samples a batch of tasks from the underlying meta-dataset. |
| `SampleOne` | Samples a single episode from the underlying meta-dataset. |
| `SetSeed(Int32)` | Sets the random seed for reproducible sampling. |
| `UpdateWithFeedback(IReadOnlyList<IEpisode<,,>>,IReadOnlyList<Double>)` | Updates the sampler state after observing a loss for a batch of tasks. |


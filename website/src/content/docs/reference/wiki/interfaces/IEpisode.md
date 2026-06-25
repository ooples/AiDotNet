---
title: "IEpisode<T, TInput, TOutput>"
description: "Represents a single episode in meta-learning, wrapping an `IMetaLearningTask` with additional metadata such as domain, difficulty, and timing information."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a single episode in meta-learning, wrapping an `IMetaLearningTask`
with additional metadata such as domain, difficulty, and timing information.

## For Beginners

An episode is one complete "mini-learning session" in meta-learning.
Each episode contains a task (support set + query set) along with extra information about that task,
like how hard it is and what domain it came from.

## Properties

| Property | Summary |
|:-----|:--------|
| `CreatedTimestamp` | Gets the timestamp when this episode was created. |
| `Difficulty` | Gets an optional difficulty score for this episode, typically in [0, 1]. |
| `Domain` | Gets an optional domain or category label for this episode. |
| `EpisodeId` | Gets the unique identifier for this episode. |
| `EpisodeMetadata` | Gets optional key-value metadata associated with the episode. |
| `LastLoss` | Gets or sets the loss observed on this episode during the most recent evaluation. |
| `SampleCount` | Gets or sets the number of times this episode has been sampled. |
| `Task` | Gets the underlying meta-learning task containing support and query sets. |


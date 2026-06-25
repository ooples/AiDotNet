---
title: "MePoOptions<T, TInput, TOutput>"
description: "Configuration options for the MePo (Memory Prototypes) meta-learning algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the MePo (Memory Prototypes) meta-learning algorithm.

## How It Works

MePo maintains a memory bank of gradient-space prototypes from previously seen tasks.
When adapting to a new task, it retrieves the nearest prototypes and uses them to
warm-start adaptation and regularize the inner loop toward known good trajectories.

## Properties

| Property | Summary |
|:-----|:--------|
| `MemorySize` | Maximum number of prototypes stored in the memory bank. |
| `PrototypeDim` | Dimensionality of each prototype vector (compressed gradient space). |
| `PrototypeRegWeight` | Regularization weight pulling adapted parameters toward the trajectory suggested by the nearest retrieved prototypes. |
| `RetrievalTopK` | Number of nearest prototypes to retrieve for warm-starting and regularization. |


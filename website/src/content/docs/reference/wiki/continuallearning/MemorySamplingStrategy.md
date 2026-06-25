---
title: "MemorySamplingStrategy"
description: "Memory sampling strategies for experience replay."
section: "API Reference"
---

`Enums` · `AiDotNet.ContinualLearning.Interfaces`

Memory sampling strategies for experience replay.

## For Beginners

When storing examples from previous tasks, the sampling
strategy determines how examples are selected and maintained in memory.

## Fields

| Field | Summary |
|:-----|:--------|
| `Boundary` | Boundary-focused sampling selects examples near decision boundaries. |
| `ClassBalanced` | Class-balanced sampling ensures equal representation of each class. |
| `GradientBased` | Gradient-based sample selection. |
| `Herding` | Herding-based selection picks examples closest to class means. |
| `KCenter` | K-Center coreset selection maximizes coverage of the feature space. |
| `Random` | Random uniform sampling from the dataset. |
| `Reservoir` | Reservoir sampling - uniform random selection. |
| `RingBuffer` | Ring buffer - FIFO replacement. |


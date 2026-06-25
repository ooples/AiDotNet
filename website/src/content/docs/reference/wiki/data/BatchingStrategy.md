---
title: "BatchingStrategy"
description: "Defines strategies for batching tasks in meta-learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Structures`

Defines strategies for batching tasks in meta-learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adaptive` | Adaptive batching based on memory constraints and task complexity. |
| `CurriculumAware` | Curriculum-aware batching following learning progression. |
| `DifficultyBased` | Group tasks by similar difficulty levels. |
| `DomainBalanced` | Balanced sampling across different task domains. |
| `HardNegativeMining` | Hard negative mining for challenging batches. |
| `MultiResolution` | Multi-resolution batching with varying K-shot configurations. |
| `SimilarityBased` | Group similar tasks together for specialized training. |
| `Uniform` | Random uniform sampling of tasks. |


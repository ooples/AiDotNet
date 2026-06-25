---
title: "MPTSOptions<T, TInput, TOutput>"
description: "Configuration options for the MPTS (Meta-learning with Progressive Task-Specific adaptation) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the MPTS (Meta-learning with Progressive Task-Specific adaptation) algorithm.

## How It Works

MPTS groups model parameters into blocks and learns per-block adaptation priorities.
High-priority groups are adapted in early inner loop steps, while lower-priority groups
are gradually unfrozen as adaptation progresses — a progressive unfreezing strategy
that reduces overfitting on small support sets.

## Properties

| Property | Summary |
|:-----|:--------|
| `GroupRegWeight` | L2 regularization weight between groups to encourage coherent adaptation. |
| `NumParamGroups` | Number of parameter groups for progressive adaptation. |
| `PriorityDecayRate` | Rate at which group activation decays from high to low priority. |


---
title: "DynamicTaskSamplingOptions<T, TInput, TOutput>"
description: "Configuration options for the DynamicTaskSampling algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the DynamicTaskSampling algorithm.

## How It Works

DynamicTaskSampling maintains per-task difficulty estimates and uses them to reweight
the meta-gradient. Tasks with higher difficulty (loss after adaptation) receive higher
weights via a softmax-based difficulty-proportional weighting, focusing meta-learning
on tasks that the model struggles with most.

## Properties

| Property | Summary |
|:-----|:--------|
| `DifficultyDecay` | EMA decay for tracking per-task difficulty (running mean of query losses). |
| `ExplorationCoeff` | Exploration coefficient (UCB-style) added to difficulty for exploration. |
| `TaskTemperature` | Temperature for difficulty-weighted gradient scaling. |


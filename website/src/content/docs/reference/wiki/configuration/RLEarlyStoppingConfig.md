---
title: "RLEarlyStoppingConfig<T>"
description: "Configuration for early stopping during RL training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration for early stopping during RL training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RLEarlyStoppingConfig` | Creates a new instance with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MinImprovement` | Minimum improvement to reset patience counter. |
| `PatienceEpisodes` | Stop if no improvement for this many episodes. |
| `RewardThreshold` | Stop training if average reward exceeds this threshold. |


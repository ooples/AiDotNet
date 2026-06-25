---
title: "ReinforcementLearningOptions<T>"
description: "Configuration options for reinforcement learning agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ReinforcementLearning.Agents`

Configuration options for reinforcement learning agents.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Batch size for training updates. |
| `DiscountFactor` | Discount factor (gamma) for future rewards. |
| `EpsilonDecay` | Exploration decay rate. |
| `EpsilonEnd` | Final exploration rate. |
| `EpsilonStart` | Initial exploration rate (for epsilon-greedy policies). |
| `LearningRate` | Learning rate for gradient updates. |
| `LossFunction` | Loss function to use for training. |
| `MaxGradientNorm` | Maximum gradient norm for clipping (0 = no clipping). |
| `ReplayBufferSize` | Size of the replay buffer (if applicable). |
| `TargetUpdateFrequency` | Frequency of target network updates (if applicable). |
| `UsePrioritizedReplay` | Whether to use prioritized experience replay. |
| `WarmupSteps` | Number of warmup steps before training. |


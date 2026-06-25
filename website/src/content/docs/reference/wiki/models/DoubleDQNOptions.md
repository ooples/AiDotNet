---
title: "DoubleDQNOptions<T>"
description: "Configuration options for Double DQN agent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Double DQN agent.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Number of discrete actions available to the agent. |
| `BatchSize` | Number of experiences sampled per training update. |
| `DiscountFactor` | Discount factor (gamma) for future rewards. |
| `EpsilonDecay` | Multiplicative decay factor applied to epsilon each episode. |
| `EpsilonEnd` | Final exploration rate. |
| `EpsilonStart` | Initial exploration rate for epsilon-greedy policy. |
| `HiddenLayers` | Hidden layer sizes for the Q-network. |
| `LearningRate` | Learning rate for gradient updates. |
| `LossFunction` | Loss function for training. |
| `ReplayBufferSize` | Maximum number of experiences stored in the replay buffer. |
| `StateSize` | Dimension of the environment state vector. |
| `TargetUpdateFrequency` | Number of steps between target network updates. |
| `WarmupSteps` | Number of random steps before training begins. |


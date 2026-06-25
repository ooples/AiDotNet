---
title: "SACOptions<T>"
description: "Configuration options for Soft Actor-Critic (SAC) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Soft Actor-Critic (SAC) agents.

## For Beginners

SAC is one of the best algorithms for continuous control (like robot movement).

Key innovations:

- **Maximum Entropy**: Encourages exploration by being "random on purpose"
- **Off-Policy**: Learns from old experiences (sample efficient)
- **Twin Q-Networks**: Uses two Q-functions to prevent overestimation
- **Automatic Tuning**: Adjusts exploration automatically

Think of it like learning to drive while staying diverse in your driving style -
you don't just learn one way to drive, you stay flexible and adaptable.

Used by: Robotic manipulation, dexterous control, autonomous systems

## How It Works

SAC is a state-of-the-art off-policy actor-critic algorithm that combines maximum
entropy RL with stable off-policy learning. It's particularly effective for
continuous control tasks and is known for excellent sample efficiency and robustness.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the continuous action space. |
| `AlphaLearningRate` | Learning rate for temperature parameter (alpha). |
| `AutoTuneTemperature` | Whether to automatically tune the temperature parameter. |
| `BatchSize` | Mini-batch size for training. |
| `DiscountFactor` | Discount factor (gamma) for future rewards. |
| `GradientSteps` | Number of gradient steps per environment step. |
| `InitialTemperature` | Initial temperature (alpha) for entropy regularization. |
| `PolicyHiddenLayers` | Hidden layer sizes for policy network. |
| `PolicyLearningRate` | Learning rate for policy network. |
| `QHiddenLayers` | Hidden layer sizes for Q-networks. |
| `QLearningRate` | Learning rate for Q-networks. |
| `QLossFunction` | Loss function for Q-networks (typically MSE). |
| `ReplayBufferSize` | Capacity of the experience replay buffer. |
| `StateSize` | Size of the state observation space. |
| `TargetEntropy` | Target entropy for automatic temperature tuning. |
| `TargetUpdateTau` | Soft target update coefficient (tau). |
| `WarmupSteps` | Number of warmup steps before starting training. |


---
title: "PPOOptions<T>"
description: "Configuration options for Proximal Policy Optimization (PPO) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Proximal Policy Optimization (PPO) agents.

## For Beginners

PPO learns a policy (strategy for choosing actions) by making careful, controlled updates.
It's like learning to drive - you make small adjustments to your steering rather than
jerking the wheel wildly. This makes learning stable and efficient.

Key features:

- **Actor-Critic**: Learns both a policy (actor) and value function (critic)
- **Clipped Updates**: Prevents too-large changes that could break learning
- **GAE**: Generalized Advantage Estimation for better gradient estimates
- **Multi-Epoch**: Reuses collected experience multiple times

Famous for: OpenAI's ChatGPT uses PPO for RLHF (Reinforcement Learning from Human Feedback)

## How It Works

PPO is a state-of-the-art policy gradient algorithm that achieves a balance between
sample efficiency, simplicity, and reliability. It uses a clipped surrogate objective
to prevent destructively large policy updates.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Number of possible actions (discrete) or action dimensions (continuous). |
| `ClipEpsilon` | PPO clipping parameter (epsilon). |
| `DiscountFactor` | Discount factor (gamma) for future rewards. |
| `EntropyCoefficient` | Entropy coefficient for exploration. |
| `GaeLambda` | GAE (Generalized Advantage Estimation) lambda parameter. |
| `IsContinuous` | Whether the action space is continuous (true) or discrete (false). |
| `MaxGradNorm` | Maximum gradient norm for gradient clipping. |
| `MiniBatchSize` | Mini-batch size for training. |
| `PolicyHiddenLayers` | Hidden layer sizes for policy network. |
| `PolicyLearningRate` | Learning rate for the policy network. |
| `StateSize` | Size of the state observation space. |
| `StepsPerUpdate` | Number of steps to collect before each training update. |
| `TrainingEpochs` | Number of epochs to train on collected data. |
| `ValueHiddenLayers` | Hidden layer sizes for value network. |
| `ValueLearningRate` | Learning rate for the value network. |
| `ValueLossCoefficient` | Value function loss coefficient. |
| `ValueLossFunction` | Loss function for value network (typically MSE). |


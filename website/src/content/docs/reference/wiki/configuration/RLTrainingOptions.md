---
title: "RLTrainingOptions<T>"
description: "Configuration options for reinforcement learning training loops via AiModelBuilder."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for reinforcement learning training loops via AiModelBuilder.

## For Beginners

Reinforcement learning trains an agent through trial and error
in an environment. This options class lets you customize every aspect of that training process:

- How many episodes to run
- How to explore vs exploit
- How to store and sample experiences
- When to receive progress updates

**Quick Start Example:**

## How It Works

This class provides comprehensive configuration for RL training loops, following industry-standard
patterns from libraries like Stable-Baselines3, RLlib, and CleanRL.

**Note:** This class is for configuring the training loop (episodes, steps, callbacks).
For agent-specific options (learning rate, discount factor), see each agent's options class.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for sampling from the replay buffer. |
| `CheckpointConfig` | Gets or sets the checkpoint configuration for saving models during training. |
| `EarlyStoppingConfig` | Gets or sets the early stopping configuration. |
| `Environment` | Gets or sets the environment for the agent to interact with. |
| `Episodes` | Gets or sets the number of episodes to train for. |
| `EvaluationConfig` | Gets or sets the evaluation configuration for assessing agent performance during training. |
| `ExplorationSchedule` | Gets or sets the exploration schedule configuration. |
| `ExplorationStrategy` | Gets or sets the optional exploration strategy to use during training. |
| `GradientSteps` | Gets or sets the number of gradient steps per training update. |
| `LogFrequency` | Gets or sets how often to log progress (every N episodes). |
| `MaxStepsPerEpisode` | Gets or sets the maximum steps per episode to prevent infinite loops. |
| `NormalizeObservations` | Gets or sets whether to normalize observations. |
| `NormalizeRewards` | Gets or sets whether to normalize rewards. |
| `OnEpisodeComplete` | Gets or sets the callback invoked after each episode completes. |
| `OnStepComplete` | Gets or sets the callback invoked after each training step. |
| `OnTrainingComplete` | Gets or sets the callback invoked when training ends. |
| `OnTrainingStart` | Gets or sets the callback invoked when training starts. |
| `PrioritizedReplayConfig` | Gets or sets the prioritized replay configuration. |
| `ReplayBuffer` | Gets or sets the optional replay buffer for experience storage. |
| `RewardClipping` | Gets or sets the reward clipping bounds. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `TargetNetworkConfig` | Gets or sets the target network configuration for DQN-family algorithms. |
| `TrainFrequency` | Gets or sets the frequency of training updates (every N steps). |
| `UsePrioritizedReplay` | Gets or sets whether to use prioritized experience replay. |
| `WarmupSteps` | Gets or sets the number of initial random steps before training begins. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Default(IEnvironment<>)` | Creates default options with sensible values for most use cases. |


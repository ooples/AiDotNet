---
title: "DQNAgent<T>"
description: "Deep Q-Network (DQN) agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DQN`

Deep Q-Network (DQN) agent for reinforcement learning.

## For Beginners

DQN learns to play games (or solve problems) by learning how valuable each action is in each situation.
It uses a neural network to estimate these "Q-values" - essentially, expected future rewards.

The agent:

- Sees the current state (like game screen)
- Evaluates each possible action using its Q-network
- Picks the action with highest Q-value (with some random exploration)
- Learns from past experiences stored in memory

Famous for: Learning to play Atari games from pixels (DeepMind, 2015)

## How It Works

DQN is a landmark algorithm that combined Q-learning with deep neural networks, enabling RL
to scale to high-dimensional state spaces. It introduced two key innovations:

1. Experience Replay: Breaks temporal correlations by training on random past experiences
2. Target Network: Provides stable Q-value targets by using a slowly-updating copy

**Reference:**
Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DQNAgent` | Initializes a new instance with default settings. |
| `DQNAgent(DQNOptions<>)` | Initializes a new instance of the DQNAgent class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` |  |
| `Deserialize(Byte[])` |  |
| `GetMetrics` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `LoadModel(String)` |  |
| `SaveModel(String)` |  |
| `SelectAction(Vector<>,Boolean)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` |  |
| `Train` |  |


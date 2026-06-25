---
title: "DuelingDQNAgent<T>"
description: "Dueling Deep Q-Network agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DuelingDQN`

Dueling Deep Q-Network agent for reinforcement learning.

## For Beginners

Dueling DQN splits Q-values into two parts:

- **Value V(s)**: How good is this state overall?
- **Advantage A(s,a)**: How much better is action 'a' compared to average?
- **Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))**

This is powerful because:

- The agent learns state values even when actions don't matter much
- Faster learning in scenarios where action choice rarely matters
- Better generalization across similar states

Example: In a car driving game, being on the road is valuable regardless of whether
you accelerate slightly or not. Dueling DQN learns "being on road = good" separately
from "how much to accelerate".

## How It Works

Dueling DQN separates the estimation of state value V(s) and action advantages A(s,a),
allowing the network to learn which states are valuable without having to learn the
effect of each action for each state. This architecture is particularly effective when
many actions do not affect the state in a relevant way.

**Reference:**
Wang et al., "Dueling Network Architectures for Deep RL", 2016.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DuelingDQNAgent` | Initializes a new instance with default settings. |

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


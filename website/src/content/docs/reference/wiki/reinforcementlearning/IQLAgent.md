---
title: "IQLAgent<T>"
description: "Implicit Q-Learning (IQL) agent for offline reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.IQL`

Implicit Q-Learning (IQL) agent for offline reinforcement learning.

## For Beginners

IQL is an offline RL algorithm that learns from fixed datasets.
It uses a clever statistical technique (expectile regression) to avoid
overestimating values of unseen actions.

Key features:

- **Expectile Regression**: Asymmetric loss that focuses on upper quantiles
- **Three Networks**: V(s), Q(s,a), and π(a|s)
- **Simpler than CQL**: No conservative penalties or Lagrangian multipliers
- **Advantage-Weighted Regression**: Extracts policy from Q and V functions

Think of expectiles like percentiles - focusing on "typically good" outcomes
rather than "best possible" outcomes helps avoid overoptimism.

Advantages:

- Simpler hyperparameter tuning than CQL
- Often more stable
- Good for offline datasets with diverse quality

## How It Works

IQL uses expectile regression to learn a value function that focuses on
high-return trajectories, enabling effective offline policy learning without
explicit conservative penalties like CQL.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IQLAgent` | Initializes a new instance with default settings. |

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
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `LoadModel(String)` |  |
| `LoadOfflineData(List<ValueTuple<Vector<>,Vector<>,,Vector<>,Boolean>>)` | Load offline dataset into the replay buffer. |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |


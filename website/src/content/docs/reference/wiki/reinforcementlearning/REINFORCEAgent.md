---
title: "REINFORCEAgent<T>"
description: "REINFORCE (Monte Carlo Policy Gradient) agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.REINFORCE`

REINFORCE (Monte Carlo Policy Gradient) agent for reinforcement learning.

## For Beginners

REINFORCE is the "hello world" of policy gradient methods. The algorithm is beautifully simple:

1. Play an entire episode
2. Calculate total rewards for each action
3. Make good actions more likely, bad actions less likely

Think of it like learning to play a game:

- You play a round
- At the end, you see your score
- You adjust your strategy to do better next time

**Pros**: Simple, works for any problem, easy to understand
**Cons**: High variance, slow learning, requires complete episodes

Modern algorithms like PPO and A2C improve on REINFORCE's core ideas.

## How It Works

REINFORCE is the simplest and most fundamental policy gradient algorithm. It directly
optimizes the policy by following the gradient of expected returns. Despite its simplicity,
it forms the foundation for many modern RL algorithms.

**Reference:**
Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist RL."

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


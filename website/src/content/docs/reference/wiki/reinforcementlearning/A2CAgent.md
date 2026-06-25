---
title: "A2CAgent<T>"
description: "Advantage Actor-Critic (A2C) agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.A2C`

Advantage Actor-Critic (A2C) agent for reinforcement learning.

## For Beginners

A2C learns two networks simultaneously:

- **Actor**: Decides which action to take (policy)
- **Critic**: Evaluates how good the current state is (value function)

The critic helps the actor learn faster by providing better feedback than rewards alone.
Think of it like having a coach (critic) give you targeted advice instead of just
saying "good" or "bad" after the game ends.

A2C is simpler than PPO but still very effective. Good starting point for actor-critic methods.

## How It Works

A2C is a synchronous, simpler version of A3C that combines policy gradients with value
function learning. It's the foundation for many modern RL algorithms including PPO.

**Reference:**
Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", 2016 (describes A3C, A2C is the synchronous version).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `A2CAgent` | Initializes a new instance with default settings. |

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


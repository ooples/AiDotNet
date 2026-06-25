---
title: "PPOAgent<T>"
description: "Proximal Policy Optimization (PPO) agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.PPO`

Proximal Policy Optimization (PPO) agent for reinforcement learning.

## For Beginners

PPO is one of the most popular RL algorithms today. It's used by:

- OpenAI's ChatGPT (for RLHF training)
- Many robotics systems
- Game AI (including Dota 2 bots)

Key idea: Make small, safe policy improvements by clipping updates.
Think of it like driving - small steering adjustments work better than jerking the wheel.

PPO learns two things:

- A policy (actor): What action to take in each state
- A value function (critic): How good each state is

The critic helps the actor learn more efficiently.

## How It Works

PPO is a policy gradient method that uses a clipped surrogate objective to enable
multiple epochs of minibatch updates without destructively large policy changes.
It achieves state-of-the-art performance across many RL benchmarks while being
simpler and more robust than methods like TRPO.

**Reference:**
Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PPOAgent` | Initializes a new instance with default settings. |
| `PPOAgent(PPOOptions<>)` | Initializes a new instance of the PPOAgent class. |

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


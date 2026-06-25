---
title: "SACAgent<T>"
description: "Soft Actor-Critic (SAC) agent for continuous control reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.SAC`

Soft Actor-Critic (SAC) agent for continuous control reinforcement learning.

## For Beginners

SAC is one of the best algorithms for continuous control (robot movement, etc.).

Key innovations:

- **Maximum Entropy**: Learns to be both effective AND diverse
- **Twin Q-Networks**: Two critics prevent overestimation
- **Automatic Tuning**: Adjusts exploration automatically
- **Off-Policy**: Very sample efficient

Think of it like learning to drive: you want to reach your destination (high reward)
but also maintain flexibility in how you drive (high entropy). This makes the policy
more robust and adaptable.

Used by: Boston Dynamics robots, autonomous vehicles, dexterous manipulation

## How It Works

SAC is a state-of-the-art off-policy actor-critic algorithm that achieves high sample
efficiency and robustness by incorporating maximum entropy reinforcement learning.
It's particularly effective for continuous control tasks.

**Reference:**
Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor", 2018.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SACAgent` | Initializes a new instance with default settings. |
| `SACAgent(SACOptions<>)` | Initializes a new instance of the SACAgent class. |

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
| `Train(Vector<>,Vector<>)` |  |


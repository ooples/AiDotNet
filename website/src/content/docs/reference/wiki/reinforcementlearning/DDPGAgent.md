---
title: "DDPGAgent<T>"
description: "Deep Deterministic Policy Gradient (DDPG) agent for continuous control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DDPG`

Deep Deterministic Policy Gradient (DDPG) agent for continuous control.

## For Beginners

DDPG is perfect for controlling things that need precise, continuous adjustments like:

- Robot arm angles (not just "left" or "right", but exact degrees)
- Car steering and acceleration (smooth continuous values)
- Temperature control, volume levels, etc.

Key components:

- **Actor**: Learns the best action to take (deterministic policy)
- **Critic**: Evaluates how good that action is (Q-value)
- **Target Networks**: Stable copies for training
- **Exploration Noise**: Adds randomness during training for exploration

Think of it like learning to drive: the actor is your decision-making (how much to
turn the wheel), the critic is your judgment (was that a good turn?), and noise
is trying slight variations to discover better techniques.

## How It Works

DDPG is an actor-critic algorithm designed for continuous action spaces. It learns
a deterministic policy (actor) and uses an off-policy approach with experience replay
and target networks, extending DQN ideas to continuous control.

**Reference:**
Lillicrap et al., "Continuous control with deep reinforcement learning", 2015.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDPGAgent` | Initializes a new instance with default settings. |

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


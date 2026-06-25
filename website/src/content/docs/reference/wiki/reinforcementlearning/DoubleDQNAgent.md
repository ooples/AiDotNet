---
title: "DoubleDQNAgent<T>"
description: "Double Deep Q-Network (Double DQN) agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DoubleDQN`

Double Deep Q-Network (Double DQN) agent for reinforcement learning.

## For Beginners

Standard DQN tends to overestimate Q-values because it uses the same network to both
select and evaluate actions (max operator causes positive bias).

Double DQN fixes this by:

- Using online network to SELECT the best action
- Using target network to EVALUATE that action's value

Think of it like getting a second opinion: one expert picks what looks best,
another expert judges its actual value. This reduces overoptimistic estimates.

**Key Improvement**: More stable learning, better performance, especially when
there's noise or stochasticity in the environment.

## How It Works

Double DQN addresses the overestimation bias in standard DQN by decoupling action
selection from action evaluation. It uses the online network to select actions and
the target network to evaluate them, leading to more accurate Q-value estimates.

**Reference:**
van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", 2015.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DoubleDQNAgent` | Initializes a new instance with default options. |
| `DoubleDQNAgent(DoubleDQNOptions<>)` | Initializes a new instance of the DoubleDQNAgent class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Not supported for DoubleDQNAgent. |
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


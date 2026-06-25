---
title: "DeepReinforcementLearningAgentBase<T>"
description: "Base class for deep reinforcement learning agents that use neural networks as function approximators."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ReinforcementLearning.Agents`

Base class for deep reinforcement learning agents that use neural networks as function approximators.

## For Beginners

This is the base class for modern "deep" RL agents.

Deep RL uses neural networks to approximate the policy and/or value functions, enabling
agents to handle high-dimensional state spaces (like images) and complex decision problems.

Classical RL methods (tabular Q-learning, linear approximation) inherit directly from
ReinforcementLearningAgentBase, while deep RL methods (DQN, PPO, A3C, etc.) inherit from
this class which adds neural network support.

Examples of deep RL algorithms:

- DQN family (DQN, Double DQN, Rainbow)
- Policy gradient methods (PPO, TRPO, A3C)
- Actor-Critic methods (SAC, TD3, DDPG)
- Model-based methods (Dreamer, MuZero, World Models)
- Transformer-based methods (Decision Transformer)

## How It Works

This class extends ReinforcementLearningAgentBase to provide specific support for neural network-based
RL algorithms. It manages neural network instances and provides infrastructure for deep RL methods.

**Auto-Compile:** Policy inference goes through the standard neural-network path,
which is auto-compiled by Tensors' AutoTracer once the input-shape pattern repeats. No
explicit compile call is required. Users can opt out via
`TensorCodecOptions.Current.EnableCompilation = false`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepReinforcementLearningAgentBase(ReinforcementLearningOptions<>)` | Initializes a new instance of the DeepReinforcementLearningAgentBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for hardware-accelerated vector operations. |
| `ParameterCount` | Gets the total number of trainable parameters across all networks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Disposes of resources used by the agent, including neural networks. |
| `GetPolicyNetworkForJit` | Gets the policy network used for action selection. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Networks` | The neural network(s) used by this agent for function approximation. |


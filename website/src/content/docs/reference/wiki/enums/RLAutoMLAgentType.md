---
title: "RLAutoMLAgentType"
description: "Defines which reinforcement learning agent families can be explored by AutoML."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines which reinforcement learning agent families can be explored by AutoML.

## For Beginners

Different RL agents are better suited for different problems:

- `DQN` is popular for discrete action spaces (like left/right).
- `PPO` is a strong general-purpose agent (discrete or continuous).
- `A2C` is a simple actor-critic baseline.
- `DDPG` and `SAC` are commonly used for continuous control.

## How It Works

This enum is used by facade configuration options to select which RL agent types AutoML is allowed to try.

## Fields

| Field | Summary |
|:-----|:--------|
| `A2C` | Advantage Actor-Critic (A2C) for discrete or continuous control. |
| `DDPG` | Deep Deterministic Policy Gradient (DDPG) for continuous control. |
| `DQN` | Deep Q-Network (DQN) for discrete action spaces. |
| `PPO` | Proximal Policy Optimization (PPO) for discrete or continuous control. |
| `SAC` | Soft Actor-Critic (SAC) for continuous control. |


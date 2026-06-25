---
title: "MADDPGAgent<T>"
description: "Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent for cooperative and competitive multi-agent reinforcement learning with continuous action spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.MADDPG`

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent for cooperative
and competitive multi-agent reinforcement learning with continuous action spaces.

## For Beginners

MADDPG enables multiple agents to learn together in shared environments.
During training, critics can "see" all agents' actions (centralized),
but during execution, each agent acts independently (decentralized).

Key features:

- **Centralized Critics**: Observe all agents during training
- **Decentralized Actors**: Independent policies per agent
- **Continuous Actions**: Based on DDPG
- **Cooperative or Competitive**: Handles both settings

Think of it like: Team sports where players practice together seeing
everyone's moves, but during games each makes independent decisions.

Examples: Robot swarms, traffic control, multi-player games

## How It Works

MADDPG extends DDPG to multi-agent settings with centralized training
and decentralized execution.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Not supported for MADDPGAgent. |
| `Clone` | Creates a deep copy of this MADDPG agent including all trained network weights. |
| `Deserialize(Byte[])` | Deserializes a MADDPG agent from a byte array. |
| `GetOptions` |  |
| `LoadModel(String)` | Loads a trained model from a file. |
| `SaveModel(String)` | Saves the trained model to a file. |
| `Serialize` | Serializes the MADDPG agent to a byte array. |
| `StoreMultiAgentExperience(List<Vector<>>,List<Vector<>>,List<>,List<Vector<>>,Boolean)` | Store multi-agent experience with per-agent reward tracking. |


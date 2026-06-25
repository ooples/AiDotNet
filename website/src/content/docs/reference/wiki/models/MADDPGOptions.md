---
title: "MADDPGOptions<T>"
description: "Configuration options for Multi-Agent DDPG (MADDPG) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Multi-Agent DDPG (MADDPG) agents.

## For Beginners

MADDPG allows multiple agents to learn together in shared environments.
During training, agents can "see" what others are doing (centralized critics),
but during execution, each agent acts independently (decentralized actors).

Key features:

- **Centralized Training**: Critics see all agents' observations and actions
- **Decentralized Execution**: Actors only use their own observations
- **Continuous Actions**: Based on DDPG for continuous control
- **Cooperative or Competitive**: Works for both settings

Think of it like: Team sports where players practice together (centralized)
but during the game each player makes their own decisions (decentralized).

Examples: Robot coordination, traffic control, multi-player games

## How It Works

MADDPG extends DDPG to multi-agent settings with centralized training and
decentralized execution. Critics observe all agents during training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MADDPGOptions` | Initializes a new instance with default values. |
| `MADDPGOptions(MADDPGOptions<>)` | Initializes a new instance by copying values from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | The per-agent action vector size. |
| `ActorHiddenLayers` | Hidden layer sizes for each agent's actor network. |
| `ActorLearningRate` | Learning rate for actor network updates. |
| `CriticHiddenLayers` | Hidden layer sizes for each agent's critic network. |
| `CriticLearningRate` | Learning rate for critic network updates. |
| `ExplorationNoise` | Standard deviation of Gaussian noise added for exploration. |
| `NumAgents` | The number of agents trained jointly. |
| `Optimizer` | The optimizer used for updating network parameters. |
| `StateSize` | The per-agent observation vector size. |
| `TargetUpdateTau` | Soft update coefficient for target networks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that required properties are set. |


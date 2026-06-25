---
title: "QMIXOptions<T>"
description: "Configuration options for QMIX agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for QMIX agents.

## For Beginners

QMIX solves multi-agent problems by learning individual Q-values for each agent,
then combining them with a "mixing network" that ensures the team's joint action
is consistent with individual actions.

Key features:

- **Value Factorization**: Decomposes team value into agent values
- **Mixing Network**: Combines agent Q-values monotonically
- **Decentralized Execution**: Each agent acts independently
- **Discrete Actions**: Value-based method for discrete action spaces

Think of it like: Each team member estimates their contribution, and a coach
(mixing network) combines these to determine the team's overall performance.

Famous for: StarCraft micromanagement, cooperative games

## How It Works

QMIX factorizes joint action-values into per-agent values using a mixing network.
This enables decentralized execution while maintaining centralized training.

## Properties

| Property | Summary |
|:-----|:--------|
| `Optimizer` | The optimizer used for updating network parameters. |


---
title: "SARSAOptions<T>"
description: "Configuration options for SARSA agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for SARSA agents.

## For Beginners

SARSA is more conservative than Q-Learning because it learns from actions
it actually takes (including exploratory ones). This makes it safer in
environments where bad actions can be catastrophic.

Classic example: **Cliff Walking**

- Q-Learning learns the shortest path (risky, close to cliff)
- SARSA learns a safer path (further from cliff)

Use SARSA when:

- Safety matters during training
- You want to learn a safe policy
- Environment has dangerous states

Use Q-Learning when:

- You want the optimal policy
- Safety during training doesn't matter
- You can afford exploratory mistakes

## How It Works

SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm.
Unlike Q-Learning, it updates based on the action actually taken.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the action space (number of possible actions). |
| `StateSize` | Size of the state space (number of state features). |


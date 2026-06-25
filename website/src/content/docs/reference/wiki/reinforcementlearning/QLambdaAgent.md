---
title: "QLambdaAgent<T>"
description: "Q(lambda) agent that combines Q-learning with eligibility traces for faster credit assignment in tabular reinforcement learning environments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.EligibilityTraces`

Q(lambda) agent that combines Q-learning with eligibility traces for faster credit assignment
in tabular reinforcement learning environments.

## For Beginners

In basic Q-learning, only the most recent state-action pair gets updated when
a reward arrives. Q(lambda) remembers which states you recently visited and
updates all of them, like leaving breadcrumbs that fade over time.

The lambda parameter controls how far back credit is assigned:

- lambda = 0: Same as regular Q-learning (only update last step)
- lambda = 1: Update all visited states equally (Monte Carlo-like)
- lambda = 0.9: Good default, updates recent states more than older ones

This helps the agent learn much faster because rewards propagate backwards
through the trajectory in a single episode instead of requiring many episodes.

## How It Works

Q(lambda) extends Q-learning by maintaining eligibility traces that assign credit
to recently visited state-action pairs. When a reward is received, all eligible
state-action pairs are updated proportionally to their trace value, enabling
faster learning in long-horizon tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QLambdaAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


---
title: "TabularQLearningOptions<T>"
description: "Configuration options for Tabular Q-Learning agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Tabular Q-Learning agents.

## For Beginners

This is the simplest form of Q-Learning where we literally maintain a table.
Each row is a state, each column is an action, and the cells contain Q-values.

Best for:

- Small discrete state spaces (e.g., 10x10 grid world)
- Discrete action spaces
- Learning exact optimal policies
- Understanding RL fundamentals

Not suitable for:

- Continuous states (infinitely many states)
- Large state spaces (millions of states)
- High-dimensional observations (images, etc.)

## How It Works

Tabular Q-Learning maintains a lookup table of Q-values for discrete
state-action pairs. No neural networks or function approximation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the action space (number of possible actions). |
| `StateSize` | Size of the state space (number of state features). |


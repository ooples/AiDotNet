---
title: "FirstVisitMonteCarloAgent<T>"
description: "First-Visit Monte Carlo agent for episodic tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.MonteCarlo`

First-Visit Monte Carlo agent for episodic tasks.

## For Beginners

Monte Carlo methods learn from complete episodes. They wait until the
episode ends, then update Q-values based on the actual returns received.

Unlike TD methods (Q-Learning, SARSA), MC methods:

- **Wait for episode completion**: No bootstrapping
- **Use actual returns**: Not estimates
- **Model-free**: Don't need environment dynamics
- **First-visit**: Only count first occurrence of state-action

Perfect for: Episodic tasks (games with clear endings)
Not good for: Continuing tasks (no episode end)

Famous for: Foundation of RL, unbiased estimates

## How It Works

First-Visit MC estimates value functions by averaging returns following
the first visit to each state in an episode.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FirstVisitMonteCarloAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


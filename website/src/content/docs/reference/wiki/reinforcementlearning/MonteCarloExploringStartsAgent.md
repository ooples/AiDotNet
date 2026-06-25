---
title: "MonteCarloExploringStartsAgent<T>"
description: "Monte Carlo Exploring Starts agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.MonteCarlo`

Monte Carlo Exploring Starts agent for reinforcement learning.

## For Beginners

Exploring Starts solves the exploration problem by guaranteeing
every state-action pair has a chance of being the starting point. After the random start,
the agent acts greedily (always picks the best known action). This ensures all state-action
pairs are eventually tried, which is required for convergence. The downside is that random
starts may not be possible in all real-world environments.

## How It Works

Monte Carlo ES ensures exploration by starting each episode from a randomly
chosen state-action pair, then following the greedy policy thereafter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MonteCarloExploringStartsAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


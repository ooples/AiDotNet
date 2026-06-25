---
title: "EpsilonGreedyBanditAgent<T>"
description: "Epsilon-Greedy Multi-Armed Bandit agent."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Bandits`

Epsilon-Greedy Multi-Armed Bandit agent.

## For Beginners

The epsilon-greedy bandit is the simplest exploration strategy.
Imagine choosing between multiple slot machines (arms) at a casino. Most of the time
(1-epsilon), you pick the machine that has paid the best so far (exploit). But sometimes
(epsilon%), you pick a random machine to discover if there is something better (explore).
Common starting epsilon is 0.1 (10% exploration). Used in A/B testing and ad selection.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EpsilonGreedyBanditAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


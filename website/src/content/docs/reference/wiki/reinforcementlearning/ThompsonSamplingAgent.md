---
title: "ThompsonSamplingAgent<T>"
description: "Thompson Sampling (Bayesian) Multi-Armed Bandit agent."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Bandits`

Thompson Sampling (Bayesian) Multi-Armed Bandit agent.

## For Beginners

Thompson Sampling uses probability to decide which action to try.
For each arm/action, it maintains a belief distribution (typically Beta distribution) about
how good that action is. To choose an action, it draws a random sample from each distribution
and picks the arm with the highest sample. Actions the agent is uncertain about naturally get
explored more. This Bayesian approach often outperforms epsilon-greedy and UCB in practice
and is widely used in A/B testing and recommendation systems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThompsonSamplingAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


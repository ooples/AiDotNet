---
title: "TabularActorCriticAgent<T>"
description: "Tabular Actor-Critic agent combining policy and value learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.AdvancedRL`

Tabular Actor-Critic agent combining policy and value learning.

## For Beginners

Actor-Critic has two components working together:
the Actor (decides which action to take) and the Critic (evaluates how good the action was).
The Critic provides feedback to help the Actor improve, like a coach watching a player.
This tabular version stores both policy preferences and value estimates in tables.
It combines the benefits of policy-based methods (can learn stochastic policies) with
value-based methods (lower variance updates).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabularActorCriticAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


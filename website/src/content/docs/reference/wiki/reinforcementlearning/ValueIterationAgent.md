---
title: "ValueIterationAgent<T>"
description: "Value Iteration agent for reinforcement learning using dynamic programming."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DynamicProgramming`

Value Iteration agent for reinforcement learning using dynamic programming.

## For Beginners

Value Iteration repeatedly updates state values using the
Bellman optimality equation until they converge. Unlike Policy Iteration which does
full evaluation then improvement, Value Iteration does both in a single sweep - making
it simpler but potentially slower per iteration. Think of it like iteratively improving
a GPS estimate: each pass gets closer to the true shortest path. Requires a complete
model of the environment.

## How It Works

Value Iteration combines policy evaluation and improvement in a single update step,
converging to the optimal value function.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ValueIterationAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


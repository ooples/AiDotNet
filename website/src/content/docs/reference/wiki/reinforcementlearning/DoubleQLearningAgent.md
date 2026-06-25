---
title: "DoubleQLearningAgent<T>"
description: "Double Q-Learning agent using two Q-tables to reduce overestimation bias."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DoubleQLearning`

Double Q-Learning agent using two Q-tables to reduce overestimation bias.

## For Beginners

Q-Learning tends to overestimate Q-values because it uses max(Q) for both
selecting and evaluating actions. Double Q-Learning fixes this by using
two separate Q-tables and randomly switching which one is updated.

Key innovation:

- **Two Q-tables**: Q1 and Q2
- **Decorrelation**: Use Q1 to select action, Q2 to evaluate (or vice versa)
- **Reduced Bias**: Prevents overestimation from max operator

Famous for: Hado van Hasselt 2010, foundation for Double DQN

## How It Works

Double Q-Learning maintains two Q-tables and uses one to select actions
and the other to evaluate them, reducing maximization bias.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DoubleQLearningAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


---
title: "LinearSARSAAgent<T>"
description: "Linear SARSA agent using linear function approximation with on-policy learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.AdvancedRL`

Linear SARSA agent using linear function approximation with on-policy learning.

## For Beginners

Linear SARSA is like Linear Q-Learning but learns on-policy,
meaning it evaluates and improves the policy it is actually following. The name SARSA comes
from the update sequence: State, Action, Reward, next State, next Action. This makes it
safer for real-world applications because it accounts for the exploration the agent is doing,
unlike Q-Learning which assumes optimal future behavior.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearSARSAAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


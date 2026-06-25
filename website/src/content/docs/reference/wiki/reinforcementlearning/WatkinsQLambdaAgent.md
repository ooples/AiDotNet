---
title: "WatkinsQLambdaAgent<T>"
description: "Watkins's Q(lambda) agent that combines Q-learning with eligibility traces but cuts traces when an exploratory (non-greedy) action is taken, ensuring convergence to the optimal policy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.EligibilityTraces`

Watkins's Q(lambda) agent that combines Q-learning with eligibility traces but
cuts traces when an exploratory (non-greedy) action is taken, ensuring convergence
to the optimal policy.

## For Beginners

Regular Q(lambda) has a problem: it uses eligibility traces to propagate rewards
backwards, but since Q-learning is off-policy (learns the optimal policy while
exploring), the traces can cause incorrect updates when exploratory actions are taken.

Watkins's Q(lambda) solves this by "cutting" the traces whenever the agent explores:

- Greedy action taken: traces decay normally (lambda * gamma)
- Exploratory action taken: ALL traces reset to zero

This means:

- During greedy sequences, you get fast multi-step learning (like Q(lambda))
- When exploring, you revert to safe 1-step Q-learning

Trade-off: More conservative than naive Q(lambda), but theoretically sound.

## How It Works

Watkins's Q(lambda) is a variant of Q(lambda) that resets eligibility traces to zero
whenever the agent takes a non-greedy action. This ensures that only greedy actions
contribute to the trace-based updates, maintaining the off-policy guarantee of Q-learning
while still benefiting from multi-step credit assignment during greedy sequences.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WatkinsQLambdaAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


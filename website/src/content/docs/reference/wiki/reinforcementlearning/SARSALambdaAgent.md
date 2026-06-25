---
title: "SARSALambdaAgent<T>"
description: "SARSA(lambda) agent that combines on-policy SARSA control with eligibility traces for faster credit assignment while respecting the current exploration policy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.EligibilityTraces`

SARSA(lambda) agent that combines on-policy SARSA control with eligibility traces
for faster credit assignment while respecting the current exploration policy.

## For Beginners

SARSA(lambda) combines two ideas:

- **SARSA**: Learn from the actions you actually take (on-policy, safer)
- **Eligibility traces**: Remember and update recent states when rewards arrive

This means the agent learns faster than plain SARSA (credit propagates back
through the trajectory) while still being cautious about risky actions.

The lambda parameter controls the trace decay:

- lambda = 0: Same as regular SARSA (1-step updates)
- lambda = 1: Full Monte Carlo returns (high variance)
- lambda = 0.9: Good default balance

Best for: Environments where safety matters AND episodes are long.

## How It Works

SARSA(lambda) extends SARSA by maintaining eligibility traces that propagate
temporal-difference errors backward through recently visited state-action pairs.
Unlike Q(lambda), SARSA(lambda) is on-policy: it updates based on the action
actually taken, making it safer in environments with dangerous states.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SARSALambdaAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


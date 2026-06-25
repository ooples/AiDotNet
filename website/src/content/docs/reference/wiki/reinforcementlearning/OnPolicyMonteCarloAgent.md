---
title: "OnPolicyMonteCarloAgent<T>"
description: "On-Policy Monte Carlo Control agent with epsilon-greedy exploration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.MonteCarlo`

On-Policy Monte Carlo Control agent with epsilon-greedy exploration.

## For Beginners

On-policy MC learns by evaluating the same policy it uses
to collect data. The agent follows an epsilon-greedy strategy (mostly best action,
sometimes random) and improves that exact strategy over time. Think of learning to cook
by actually cooking with your current recipe, then adjusting based on results. Simpler
than off-policy methods but cannot reuse data from previous policies.

## How It Works

On-Policy MC Control uses epsilon-greedy policy for both behavior and target,
ensuring exploration while learning the optimal policy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnPolicyMonteCarloAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


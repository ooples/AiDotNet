---
title: "NStepSARSAAgent<T>"
description: "N-step SARSA agent using multi-step bootstrapping."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.NStepSARSA`

N-step SARSA agent using multi-step bootstrapping.

## For Beginners

Instead of updating based on just the next reward (1-step SARSA), n-step methods
look ahead n steps to get better return estimates before bootstrapping.

Update: G_t = r_t+1 + γr_t+2 + ... + γ^(n-1)r_t+n + γ^n Q(s_t+n, a_t+n)

Benefits:

- **Better credit assignment**: Propagates rewards faster than 1-step
- **Lower variance**: Than full Monte Carlo
- **Flexible**: Choose n to balance bias and variance

Common values: n=3 to n=10
Famous for: Sutton & Barto's RL textbook, Chapter 7

## How It Works

N-step SARSA uses n-step returns that look ahead multiple steps before bootstrapping.
This provides a middle ground between TD (1-step) and Monte Carlo (full episode).

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


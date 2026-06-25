---
title: "ExpectedSARSAAgent<T>"
description: "Expected SARSA agent using tabular methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.ExpectedSARSA`

Expected SARSA agent using tabular methods.

## For Beginners

Expected SARSA is like SARSA but instead of using the actual next action,
it uses the average Q-value weighted by the probability of taking each action.
This reduces variance compared to SARSA.

Update: Q(s,a) ← Q(s,a) + α[r + γ Σ π(a'|s')Q(s',a') - Q(s,a)]

Benefits over SARSA:

- **Lower Variance**: Averages over actions instead of sampling
- **Off-Policy Learning**: Can learn optimal policy while exploring
- **Better Performance**: Often converges faster than SARSA

Famous for: Van Seijen et al. 2009, bridging SARSA and Q-Learning

## How It Works

Expected SARSA is a TD control algorithm that uses the expected value under
the current policy instead of sampling the next action.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpectedSARSAAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


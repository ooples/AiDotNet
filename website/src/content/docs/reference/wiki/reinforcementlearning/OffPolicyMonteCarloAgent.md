---
title: "OffPolicyMonteCarloAgent<T>"
description: "Off-Policy Monte Carlo Control agent with weighted importance sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.MonteCarlo`

Off-Policy Monte Carlo Control agent with weighted importance sampling.

## For Beginners

Off-policy MC learns the best strategy by watching someone
else play (the behavior policy), then correcting for the difference using importance
sampling weights. This is like learning optimal chess strategy by studying games played
by beginners, adjusting for their suboptimal moves. The advantage is you can reuse old
data collected under any policy, but importance sampling can introduce high variance.

## How It Works

Off-Policy MC uses importance sampling to learn an optimal policy (target)
while following a different exploratory policy (behavior).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OffPolicyMonteCarloAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |
| `SelectGreedyActionForStateKey(String)` | Shared greedy-with-tie-break selector used by both `Boolean)` and `Vector{`. |
| `StableFnv1aHash(String)` | Deterministic 32-bit FNV-1a hash. |


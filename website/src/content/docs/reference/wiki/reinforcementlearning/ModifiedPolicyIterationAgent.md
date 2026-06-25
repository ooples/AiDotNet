---
title: "ModifiedPolicyIterationAgent<T>"
description: "Modified Policy Iteration agent - hybrid of Policy Iteration and Value Iteration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.DynamicProgramming`

Modified Policy Iteration agent - hybrid of Policy Iteration and Value Iteration.

## For Beginners

Modified Policy Iteration is a middle ground between two classic
algorithms: Value Iteration (fast but less stable) and Policy Iteration (stable but slow).
Instead of fully evaluating a policy before improving it, it does a limited number of
evaluation sweeps. Think of it like proofreading a draft: you do a few passes (not infinite)
before revising. The number of evaluation sweeps controls the speed-stability trade-off.

## How It Works

Modified PI performs limited policy evaluation sweeps before improvement,
trading off between the efficiency of VI and the stability of PI.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModifiedPolicyIterationAgent` | Initializes a new instance with default options (StateSize=4, ActionSize=2). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


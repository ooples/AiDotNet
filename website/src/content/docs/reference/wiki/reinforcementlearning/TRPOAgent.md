---
title: "TRPOAgent<T>"
description: "Trust Region Policy Optimization (TRPO) agent for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.TRPO`

Trust Region Policy Optimization (TRPO) agent for reinforcement learning.

## For Beginners

TRPO is like learning carefully - it never makes changes that are "too big".
By limiting how much the policy can change (using KL divergence), it guarantees
that performance never degrades (monotonic improvement).

Key innovations:

- **Trust Region**: Constraints on policy change (KL divergence ≤ δ)
- **Monotonic Improvement**: Provable performance guarantees
- **Conjugate Gradient**: Efficient solution to constrained optimization
- **Line Search**: Ensures constraints are satisfied

Think of it like walking carefully on uncertain terrain - small, safe steps
rather than large leaps that might cause you to fall.

Famous for: OpenAI robotics, predecessor to PPO (which simplified TRPO)

## How It Works

TRPO ensures monotonic improvement by constraining policy updates within a trust region
defined by KL divergence. This prevents destructively large updates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TRPOAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |


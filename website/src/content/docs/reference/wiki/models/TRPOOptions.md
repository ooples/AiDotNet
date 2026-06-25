---
title: "TRPOOptions<T>"
description: "Configuration options for Trust Region Policy Optimization (TRPO) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Trust Region Policy Optimization (TRPO) agents.

## For Beginners

TRPO is like learning carefully - it never makes a change that's "too big".
By limiting how much the policy can change, it guarantees that performance
never gets worse (monotonic improvement).

Key features:

- **Trust Region**: Limits policy change per update (via KL divergence)
- **Monotonic Improvement**: Guarantees performance doesn't degrade
- **Conjugate Gradient**: Efficiently solves constrained optimization
- **Line Search**: Ensures constraints are satisfied

Think of it like taking small, safe steps when walking on uncertain terrain
rather than making large leaps that might cause you to fall.

Famous for: OpenAI's robotics research, predecessor to PPO

## How It Works

TRPO ensures monotonic improvement by constraining policy updates to a "trust region"
using KL divergence. This prevents destructively large updates.

## Properties

| Property | Summary |
|:-----|:--------|
| `Optimizer` | The optimizer used for updating network parameters. |


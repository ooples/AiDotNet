---
title: "LSPIOptions<T>"
description: "Configuration options for LSPI (Least-Squares Policy Iteration) agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for LSPI (Least-Squares Policy Iteration) agents.

## For Beginners

LSPI is like repeatedly asking "what's the best policy?" and "how good is it?"
until the answers stop changing. Each iteration uses LSTD to evaluate the current
policy, then improves it based on those evaluations.

Best for:

- Batch reinforcement learning
- Offline learning from fixed datasets
- Sample-efficient policy learning
- When you need guaranteed convergence

Not suitable for:

- Online/streaming scenarios
- Very large feature spaces
- Continuous action spaces
- Real-time learning requirements

## How It Works

LSPI combines least-squares methods with policy iteration. It alternates between
policy evaluation (using LSTDQ) and policy improvement, iteratively refining
the policy until convergence.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the action space (number of possible actions). |
| `ConvergenceThreshold` | Weight change threshold for determining convergence. |
| `FeatureSize` | Number of features in the state representation. |
| `MaxIterations` | Maximum number of policy iteration steps before stopping. |
| `RegularizationParam` | Regularization parameter to prevent overfitting and ensure numerical stability. |


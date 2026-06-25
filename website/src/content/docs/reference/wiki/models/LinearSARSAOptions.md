---
title: "LinearSARSAOptions<T>"
description: "Configuration options for Linear SARSA agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Linear SARSA agents.

## For Beginners

Linear SARSA is the on-policy version of Linear Q-Learning. It learns about
the policy it's currently following, rather than the optimal policy. This makes
it safer in risky environments where exploration could be dangerous.

Best for:

- Medium-sized continuous state spaces
- Risky environments (cliff walking, robotics)
- More conservative, safe learning
- Feature-based state representations

Not suitable for:

- Very small discrete states (use tabular SARSA)
- When fastest convergence is needed (use Q-learning)
- Highly non-linear problems (use neural networks)

## How It Works

Linear SARSA uses linear function approximation for on-policy learning.
Unlike Linear Q-Learning (off-policy), SARSA updates based on the action
actually taken by the current policy, making it more conservative.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the action space (number of possible actions). |
| `FeatureSize` | Number of features in the state representation. |


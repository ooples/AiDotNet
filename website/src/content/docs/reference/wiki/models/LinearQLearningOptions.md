---
title: "LinearQLearningOptions<T>"
description: "Configuration options for Linear Q-Learning agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Linear Q-Learning agents.

## For Beginners

Linear Q-Learning extends tabular Q-learning to handle larger state spaces
by using feature representations. Think of it as learning a formula instead
of memorizing every single state.

Best for:

- Medium-sized continuous state spaces
- Problems where states can be represented as feature vectors
- Faster learning than tabular methods
- Generalization across similar states

Not suitable for:

- Very small discrete states (use tabular instead)
- Highly non-linear relationships (use neural networks)
- Continuous action spaces (use actor-critic)

## How It Works

Linear Q-Learning uses linear function approximation to estimate Q-values.
Instead of maintaining a table, it learns weight vectors for each action
and computes Q(s,a) = w_a^T * φ(s) where φ(s) are state features.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the action space (number of possible actions). |
| `FeatureSize` | Number of features in the state representation. |


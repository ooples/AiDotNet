---
title: "DreamerAgent<T>"
description: "Dreamer agent for model-based reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Agents.Dreamer`

Dreamer agent for model-based reinforcement learning.

## For Beginners

Dreamer learns a "mental model" of how the environment works, then uses that
model to imagine future scenarios and plan actions - like chess players
thinking multiple moves ahead.

Key components:

- **Representation Network**: Encodes observations to latent states
- **Dynamics Model**: Predicts next latent state
- **Reward Model**: Predicts rewards
- **Value Network**: Estimates state values
- **Actor Network**: Learns policy in imagination

Think of it as: First learn physics by observation, then use that knowledge
to predict "what happens if I do X" without actually doing it.

Advantages: Sample efficient, works with images, enables planning

## How It Works

Dreamer learns a world model in latent space and uses it for planning.
It combines representation learning, dynamics modeling, and policy learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DreamerAgent` | Initializes a new instance with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` | Computes gradients for supervised learning scenarios. |
| `GetOptions` |  |


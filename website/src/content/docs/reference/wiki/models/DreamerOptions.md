---
title: "DreamerOptions<T>"
description: "Configuration options for Dreamer agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Dreamer agents.

## For Beginners

Dreamer learns a "mental model" of how the environment works, then uses that
model to imagine future scenarios and plan actions - like playing chess in your head.

Key components:

- **World Model**: Learns environment dynamics in compact latent space
- **Representation Network**: Encodes observations to latent states
- **Transition Model**: Predicts next latent state
- **Reward Model**: Predicts rewards
- **Actor-Critic**: Learns policy by imagining trajectories

Think of it like: Learning physics by observation, then using that knowledge
to predict "what happens if I do X" without actually doing it.

Advantages: Sample efficient, works with image observations, enables planning

## How It Works

Dreamer learns a world model in latent space and uses it for planning.
It combines representation learning, dynamics modeling, and policy learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `Optimizer` | The optimizer used for updating network parameters. |


---
title: "TabularActorCriticOptions<T>"
description: "Configuration options for Tabular Actor-Critic agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Tabular Actor-Critic agents.

## For Beginners

Actor-Critic is like having both a player (actor) and a coach (critic). The player tries
different strategies, and the coach provides feedback on how well they're working.

Best for:

- Small discrete state/action spaces
- Problems requiring both policy and value learning
- More stable learning than pure policy gradient
- Reducing variance in policy updates

Not suitable for:

- Continuous states (use linear/neural versions)
- Large state spaces (table becomes too big)
- High-dimensional observations

## How It Works

Tabular Actor-Critic combines policy learning (actor) with value function learning (critic)
using lookup tables. The actor learns which actions to take, while the critic evaluates
how good those actions are.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Size of the action space (number of possible actions). |
| `ActorLearningRate` | Learning rate for the actor (policy) updates. |
| `CriticLearningRate` | Learning rate for the critic (value function) updates. |
| `StateSize` | Size of the state space (number of state features). |


---
title: "IRLAgent<T>"
description: "Marker interface for reinforcement learning agents that integrate with AiModelBuilder."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Marker interface for reinforcement learning agents that integrate with AiModelBuilder.

## For Beginners

An RL agent is just a special kind of model that learns through interaction with an environment.
By implementing IFullModel, RL agents work with all of AiDotNet's existing infrastructure:

- They can be saved and loaded
- They work with the AiModelBuilder pattern
- They support serialization, cloning, etc.

The key difference is how they're trained:

- Regular models: trained on fixed datasets (x, y)
- RL agents: trained by interacting with environments and getting rewards

## How It Works

This interface extends IFullModel to ensure RL agents integrate seamlessly with AiDotNet's
existing architecture. RL agents are models where:

- TInput = Tensor<T> (state observations, though often flattened to Vector in practice)
- TOutput = Vector<T> (actions)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMetrics` | Gets current training metrics. |
| `ResetEpisode` | Resets episode-specific state (if any). |
| `SelectAction(Vector<>,Boolean)` | Selects an action given the current state observation. |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Stores an experience tuple for later learning. |
| `Train` | Performs one training step using stored experiences. |


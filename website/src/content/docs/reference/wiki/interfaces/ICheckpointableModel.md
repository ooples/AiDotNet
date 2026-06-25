---
title: "ICheckpointableModel"
description: "Defines the contract for models that support saving and loading their internal state (checkpointing)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for models that support saving and loading their internal state (checkpointing).

## For Beginners

This interface is like a "save game" feature for machine learning models.

Just as video games let you save your progress and load it later:

- SaveState: Writes the model's current state (all learned parameters) to a file
- LoadState: Reads a previously saved state back into the model

This is useful for:

- Saving the best model during training (so you can use it later)
- Resuming training if it gets interrupted
- Sharing trained models with others
- Deploying models to production systems

## How It Works

This interface enables models to save their trained parameters and internal state to persistent storage
and restore them later, which is essential for model persistence, training interruption/resumption,
and distributed training scenarios.

**Design Note:** This is a separate interface from IFullModel because not all models
support checkpointing (e.g., some stateless models or models that can't serialize their state).
By making it optional, we keep the type system honest and allow models to opt-in to checkpointing.

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadState(Stream)` | Loads the model's state (parameters and configuration) from a stream. |
| `SaveState(Stream)` | Saves the model's current state (parameters and configuration) to a stream. |


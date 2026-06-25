---
title: "LayerCheckpointState<T>"
description: "State for layer-based gradient checkpointing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Memory`

State for layer-based gradient checkpointing.

## Properties

| Property | Summary |
|:-----|:--------|
| `FinalOutput` | Final output from forward pass. |
| `Layers` | The layers used in forward pass. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears all stored state to free memory. |
| `GetActivation(Int32)` | Gets a recomputed activation if available. |
| `GetCheckpoint(Int32)` | Gets a checkpoint if available. |
| `HasActivation(Int32)` | Checks if an activation exists. |
| `HasCheckpoint(Int32)` | Checks if a checkpoint exists. |
| `SaveActivation(Int32,Tensor<>)` | Saves a recomputed activation. |
| `SaveCheckpoint(Int32,Tensor<>)` | Saves a checkpoint at the given layer index. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activations` | Recomputed activations during backward pass. |
| `_checkpoints` | Saved checkpoints at specific layer indices. |


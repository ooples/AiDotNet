---
title: "ActivationCheckpointConfig"
description: "Configuration for activation checkpointing in pipeline parallel training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.DistributedTraining`

Configuration for activation checkpointing in pipeline parallel training.

## For Beginners

During training, the forward pass must save intermediate results
(activations) so the backward pass can compute gradients. For very deep models, storing all
these activations uses enormous amounts of memory.

Activation checkpointing is like taking notes at chapter boundaries instead of every page:

- Without checkpointing: Save every activation (lots of memory, no recomputation)
- With checkpointing: Save every Nth activation, recompute the rest (less memory, more compute)

Memory savings: O(L) → O(sqrt(L)) where L = number of layers.
For 100 layers, this reduces memory from 100 activations to ~10 activations.

The trade-off is ~33% more compute time, but this enables training models that otherwise
wouldn't fit in memory.

## How It Works

Activation checkpointing (also called gradient checkpointing) trades compute for memory
by only storing activations at checkpoint layers during the forward pass. Intermediate
activations are recomputed from the nearest checkpoint during the backward pass.

**Reference:** Chen et al., "Training Deep Nets with Sublinear Memory Cost", 2016.
https://arxiv.org/abs/1604.06174

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointEveryNLayers` | Gets or sets how often to save a checkpoint (every N layers). |
| `CheckpointFirstLayer` | Gets or sets whether to checkpoint the very first layer's input. |
| `Enabled` | Gets or sets whether activation checkpointing is enabled. |
| `MaxActivationsInMemory` | Gets or sets the maximum number of activations to keep in memory simultaneously. |
| `RecomputeStrategy` | Gets or sets the recomputation strategy to use during the backward pass. |


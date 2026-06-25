---
title: "IStatefulDataLoader<T>"
description: "Extends `IDataLoader` with checkpoint/resume support for fault-tolerant training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Extends `IDataLoader` with checkpoint/resume support for fault-tolerant training.

## For Beginners

If your training crashes halfway through an epoch,
a stateful data loader can pick up exactly where it left off instead of starting
the epoch over. This saves significant time for large datasets.

## How It Works

Inspired by PyTorch's StatefulDataLoader (2025), this interface enables mid-epoch
checkpointing and exact resumption of data iteration after crashes or preemption.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetState` | Captures the current state of the data loader as a serializable object. |
| `LoadState(DataLoaderCheckpoint)` | Restores the data loader to a previously captured state. |


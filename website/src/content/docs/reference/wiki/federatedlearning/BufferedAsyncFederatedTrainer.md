---
title: "BufferedAsyncFederatedTrainer<T>"
description: "Implements FedBuff — Buffered asynchronous federated aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Trainers`

Implements FedBuff — Buffered asynchronous federated aggregation.

## For Beginners

In pure async FL, the server aggregates each client's update
as soon as it arrives. This can lead to "stale" updates from slow clients being applied
to a model that has already moved on. FedBuff adds a buffer: the server waits until
K updates arrive (from any clients), then aggregates them all at once. This balances
freshness (not too stale) with efficiency (don't wait for all clients).

## How It Works

Algorithm:

Reference: Nguyen, J., et al. (2022). "Federated Learning with Buffered
Asynchronous Aggregation." AISTATS 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BufferedAsyncFederatedTrainer(Int32,Double)` | Creates a new FedBuff trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferSize` | Gets the buffer size (K). |
| `CurrentBufferCount` | Gets the current buffer count. |
| `CurrentGlobalRound` | Gets the current global round. |
| `IsBufferReady` | Checks if the buffer is full and ready for aggregation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateBuffer` | Aggregates the buffered updates with staleness-weighted averaging. |
| `SubmitUpdate(Int32,Dictionary<String,[]>,Int32)` | Submits a client update to the buffer. |


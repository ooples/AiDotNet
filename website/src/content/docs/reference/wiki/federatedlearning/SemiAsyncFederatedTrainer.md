---
title: "SemiAsyncFederatedTrainer<T>"
description: "Implements Semi-Asynchronous Federated Learning — hybrid sync/async with periodic barriers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Trainers`

Implements Semi-Asynchronous Federated Learning — hybrid sync/async with periodic barriers.

## For Beginners

Pure synchronous FL waits for ALL clients each round (slow, wastes
time waiting for stragglers). Pure async FL processes each update immediately (fast, but stale
updates can hurt convergence). Semi-Async is the middle ground: the server accepts async updates
between synchronization barriers that occur every K rounds. During async phases, fast clients
can contribute multiple updates. At barriers, all pending updates are aggregated and a new
global model is broadcast. This balances speed with convergence quality.

## How It Works

Algorithm:

Reference: Wu, X., et al. (2023). "Semi-Asynchronous Federated Learning:
Convergence and Efficiency." IEEE TPDS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SemiAsyncFederatedTrainer(Int32,Double,Double)` | Creates a new Semi-Async FL trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AsyncLearningRate` | Gets the async learning rate. |
| `AsyncRoundsPerBarrier` | Gets the number of async rounds per barrier. |
| `CurrentRound` | Gets the current round number. |
| `PendingUpdateCount` | Gets the number of pending buffered updates. |
| `StalenessDiscount` | Gets the staleness discount factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvanceRound` | Advances the trainer to the next round. |
| `ApplyAsyncUpdates(Dictionary<String,[]>)` | Applies buffered async updates to the global model with staleness discounting. |
| `IsBarrierRound(Int32)` | Determines whether the current round is a synchronization barrier. |
| `ReceiveUpdate(Int32,Dictionary<String,[]>,Int32)` | Receives an async client update and buffers it for application. |
| `SynchronizationBarrier(Dictionary<String,[]>,Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Int32>)` | Performs a synchronization barrier: aggregates all buffered updates via weighted average. |


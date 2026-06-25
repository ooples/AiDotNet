---
title: "AsyncFedEDTrainer<T>"
description: "Implements AsyncFedED — Asynchronous FL with Entropy-Driven client scheduling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Trainers`

Implements AsyncFedED — Asynchronous FL with Entropy-Driven client scheduling.

## For Beginners

In async FL, the server doesn't wait for all clients — it processes
updates as they arrive. But which clients should train next? AsyncFedED uses an information-
theoretic approach: it estimates each client's "information gain" (how much the global model
would improve from that client's update) using entropy of the client's local loss distribution.
Clients with higher entropy (more uncertain predictions, more to learn from) are scheduled first.
This prioritizes the most informative clients, converging faster than random scheduling.

## How It Works

Scheduling:

Reference: AsyncFedED: Entropy-Driven Scheduling for Asynchronous Federated Learning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AsyncFedEDTrainer(Double,Double,Int32)` | Creates a new AsyncFedED trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientEntropies` | Gets a snapshot of the current entropy estimates for all tracked clients. |
| `ExplorationBonus` | Gets the exploration bonus. |
| `SelectionBudget` | Gets the selection budget. |
| `StalenessDecay` | Gets the staleness decay factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateWithEntropyWeights(Dictionary<Int32,Dictionary<String,[]>>,Int32)` | Aggregates client updates with entropy-weighted contributions. |
| `SelectClients(IReadOnlyCollection<Int32>,Int32)` | Selects the most informative clients for the next round. |
| `UpdateClientEntropy(Int32,Double[],Int32)` | Updates the entropy estimate for a client based on their local loss distribution. |


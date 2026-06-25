---
title: "IClientSelectionStrategy"
description: "Selects which clients participate in a federated learning round."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Selects which clients participate in a federated learning round.

## How It Works

**For Beginners:** In federated learning, the server usually does not use every client in every round.
Instead, it picks a subset of clients to reduce communication cost and handle device availability.

Different selection strategies optimize for different goals:

- Uniform: simple and fair
- Weighted: prefer clients with more data
- Stratified: ensure each group is represented
- Availability-aware: prefer clients likely to be online
- Performance-aware: prefer clients that historically help training
- Cluster-based: sample across diverse client behaviors

## Methods

| Method | Summary |
|:-----|:--------|
| `GetStrategyName` | Gets the name of the selection strategy. |
| `SelectClients(ClientSelectionRequest)` | Selects the clients to participate for the current round. |


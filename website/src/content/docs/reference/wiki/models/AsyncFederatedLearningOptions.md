---
title: "AsyncFederatedLearningOptions"
description: "Configuration options for asynchronous federated learning (FedAsync / FedBuff)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for asynchronous federated learning (FedAsync / FedBuff).

## How It Works

**For Beginners:** In synchronous federated learning, the server waits for a whole "round" of clients
before updating the global model. In asynchronous federated learning, the server can update as client
updates arrive (or in small buffers), which can reduce waiting on slow clients.

## Properties

| Property | Summary |
|:-----|:--------|
| `AsyncFedEDExplorationBonus` | Gets or sets the exploration bonus for unvisited clients in AsyncFedED scheduling. |
| `AsyncFedEDSelectionBudget` | Gets or sets the maximum clients to select per round in AsyncFedED. |
| `FedAsyncMixingRate` | Gets or sets the base mixing rate used by FedAsync. |
| `FedBuffBufferSize` | Gets or sets the buffer size for FedBuff (number of updates to accumulate before applying a server update). |
| `Mode` | Gets or sets the async mode. |
| `RejectUpdatesWithStalenessGreaterThan` | Gets or sets the maximum staleness allowed before rejecting an update. |
| `SemiAsyncRoundsPerBarrier` | Gets or sets the number of async rounds between synchronization barriers (Semi-Async mode). |
| `SimulatedMaxClientDelaySteps` | Gets or sets the maximum simulated client delay (in server steps) for in-memory async training. |
| `StalenessDecayRate` | Gets or sets the staleness decay rate used by "Exponential" and "Polynomial" weighting. |
| `StalenessWeighting` | Gets or sets the staleness weighting mode for asynchronous updates. |


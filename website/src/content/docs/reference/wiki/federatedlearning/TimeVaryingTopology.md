---
title: "TimeVaryingTopology<T>"
description: "Implements time-varying topology for decentralized federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Decentralized`

Implements time-varying topology for decentralized federated learning.

## For Beginners

In decentralized FL, the communication graph determines which
clients can talk to each other. A fixed graph can create "bottleneck" nodes and slow convergence.
Time-varying topology changes the graph each round — this accelerates mixing (spreading
information across all clients) and makes the system more robust to node failures.

## How It Works

Strategies:

Reference: Time-Varying Communication Topologies for Decentralized FL (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeVaryingTopology(TimeVaryingTopology<>.TopologyStrategy,Int32)` | Creates a new time-varying topology. |

## Properties

| Property | Summary |
|:-----|:--------|
| `RoundCounter` | Gets the current round. |
| `Strategy` | Gets the topology strategy. |
| `TopologyName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateTopology(IReadOnlyList<Int32>)` | Generates the neighbor set for each client for the current round. |
| `GetMixingWeight(Int32,Int32,Int32)` |  |
| `GetOrGenerateTopologyForRound(Int32,Int32)` | Gets or generates the topology for a specific round without mutating _roundCounter. |
| `GetPeers(Int32,Int32,Int32)` |  |


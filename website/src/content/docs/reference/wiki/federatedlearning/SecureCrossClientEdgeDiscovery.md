---
title: "SecureCrossClientEdgeDiscovery<T>"
description: "PSI-based secure discovery of cross-client edges without revealing full adjacency lists."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

PSI-based secure discovery of cross-client edges without revealing full adjacency lists.

## For Beginners

When a graph is split across clients, some edges connect nodes on
different clients. To discover these edges privately, we use a technique from Private Set
Intersection (PSI): each client hashes their border node IDs, and we compare hashes to find
matches. Only matching (shared) node IDs are revealed — each client's non-shared nodes stay private.

## How It Works

**Protocol:**

**Privacy:** Uses the Diffie-Hellman PSI approach from #538 when available.
Falls back to hash-based comparison with randomized response for differential privacy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SecureCrossClientEdgeDiscovery(Double,Int32,Boolean)` | Initializes a new instance of `SecureCrossClientEdgeDiscovery`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiscoveredEdgeCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheEdges(Int32,Int32,IReadOnlyList<ValueTuple<Int32,Int32>>)` |  |
| `DiscoverEdges(IReadOnlyList<Int32>,IReadOnlyList<Int32>)` |  |
| `GetEdges(Int32,Int32)` |  |


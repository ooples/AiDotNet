---
title: "GossipProtocol"
description: "Gossip Protocol — randomized peer-to-peer model exchange for decentralized FL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Decentralized`

Gossip Protocol — randomized peer-to-peer model exchange for decentralized FL.

## For Beginners

Like spreading a rumor — each person tells a few random friends,
who tell their friends, and eventually everyone knows. Similarly, each device shares its
model with a few random peers each round. After enough rounds, all models converge to
the same global knowledge.

## How It Works

In gossip-based aggregation, each node randomly selects K peers per round and averages
its model with theirs. Over many rounds, this converges to the global average. The
protocol is resilient to node failures since no single node is critical.

Reference: Boyd et al. (2006), "Randomized Gossip Algorithms".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GossipProtocol(Int32,Int32)` | Creates a new gossip protocol topology. |

## Properties

| Property | Summary |
|:-----|:--------|
| `TopologyName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMixingWeight(Int32,Int32,Int32)` |  |
| `GetPeers(Int32,Int32,Int32)` |  |


---
title: "IDecentralizedTopology"
description: "Interface for decentralized peer-to-peer network topologies in serverless federated learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Decentralized`

Interface for decentralized peer-to-peer network topologies in serverless federated learning.

## For Beginners

Instead of a star pattern (everyone talks to one server), decentralized
FL uses patterns like a ring (pass the model around in a circle) or gossip (randomly share
with a few neighbors). This removes the single point of failure and can be more robust.

## How It Works

In decentralized FL, there is no central server. Nodes communicate directly with peers
according to a network topology. The topology determines which nodes can exchange models
and affects convergence speed and communication costs.

## Properties

| Property | Summary |
|:-----|:--------|
| `TopologyName` | Gets the topology name for logging. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMixingWeight(Int32,Int32,Int32)` | Gets the mixing weight for a peer's contribution during aggregation. |
| `GetPeers(Int32,Int32,Int32)` | Gets the list of peer IDs that a given node should communicate with in this round. |


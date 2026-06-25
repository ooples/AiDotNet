---
title: "DecentralizedAggregator<T>"
description: "Decentralized aggregator — performs local mixing of model parameters based on topology."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Decentralized`

Decentralized aggregator — performs local mixing of model parameters based on topology.

## For Beginners

Instead of sending your model to a central server and getting back
an average, you directly blend your model with your neighbors' models. The blending weights
come from the topology (e.g., gossip, ring). After enough blending rounds, everyone's
model converges to the same global model.

## How It Works

The decentralized aggregator replaces the central server aggregation step. Each node
performs a weighted average of its own model with models received from its peers according
to the topology's mixing weights. Over rounds, this converges to the global average.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DecentralizedAggregator(IDecentralizedTopology)` | Creates a new decentralized aggregator with the specified topology. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Topology` | Gets the topology used by this aggregator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `MixWithPeers(Int32,Vector<>,Dictionary<Int32,Vector<>>,Int32,Int32)` | Performs one round of decentralized averaging for a single node. |
| `SimulateRound(Dictionary<Int32,Vector<>>,Int32)` | Simulates a full decentralized round where all nodes exchange and mix simultaneously. |


---
title: "SegmentedGossipProtocol<T>"
description: "Implements Segmented Gossip — communication-efficient gossip that exchanges model segments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Decentralized`

Implements Segmented Gossip — communication-efficient gossip that exchanges model segments.

## For Beginners

In standard gossip protocols, each pair of communicating nodes
exchanges the entire model. For large models, this is very expensive. Segmented gossip
splits the model into segments and only exchanges one segment per communication round.
Over multiple rounds, all segments get exchanged, achieving the same convergence but with
much less per-round communication.

## How It Works

Algorithm:

Reference: Bellet, A., et al. (2024). "Segmented Gossip for Communication-Efficient
Decentralized Learning." 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegmentedGossipProtocol(Int32)` | Creates a new Segmented Gossip protocol. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressionRatio` | Gets the communication compression ratio. |
| `CurrentRound` | Gets the current round number. |
| `NumSegments` | Gets the number of segments. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GossipExchange(Dictionary<String,[]>,Dictionary<String,[]>)` | Performs one round of segmented gossip between two peers. |


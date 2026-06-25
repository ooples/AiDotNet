---
title: "RingAllReduceProtocol"
description: "Ring AllReduce — communication-efficient decentralized averaging using a ring topology."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Decentralized`

Ring AllReduce — communication-efficient decentralized averaging using a ring topology.

## For Beginners

Imagine passing cards around a circle. Each person adds their card
to the pile as it passes through them. After going around twice, everyone has seen all
the cards. This is much more efficient than having everyone send their cards to one
central person.

## How It Works

In Ring AllReduce, nodes are arranged in a logical ring. Each round consists of two phases:
(1) scatter-reduce: each node sends a chunk of its data to the next node in the ring and
receives a chunk from the previous node, reducing as it goes; (2) allgather: the fully
reduced chunks are propagated around the ring. This achieves bandwidth-optimal communication
cost of 2(N-1)/N × data_size regardless of the number of nodes.

## Properties

| Property | Summary |
|:-----|:--------|
| `TopologyName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMixingWeight(Int32,Int32,Int32)` |  |
| `GetPeers(Int32,Int32,Int32)` |  |


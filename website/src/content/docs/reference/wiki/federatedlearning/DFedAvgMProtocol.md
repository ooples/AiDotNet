---
title: "DFedAvgMProtocol<T>"
description: "Implements DFedAvgM — Decentralized FedAvg with Momentum for peer-to-peer FL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Decentralized`

Implements DFedAvgM — Decentralized FedAvg with Momentum for peer-to-peer FL.

## For Beginners

In decentralized FL, there's no central server. Clients
communicate directly with their neighbors in a network graph. DFedAvgM improves on basic
decentralized averaging by adding momentum to the averaging step, which smooths out the
oscillations caused by heterogeneous data and sparse communication graphs.

## How It Works

Algorithm per round:

Reference: Sun, T., et al. (2023). "Decentralized Federated Averaging with Momentum."
TMLR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DFedAvgMProtocol(Double)` | Creates a new DFedAvgM protocol. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Momentum` | Gets the momentum coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AverageWithMomentum(Int32,Dictionary<String,[]>,Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Performs one round of decentralized averaging with momentum for a client. |


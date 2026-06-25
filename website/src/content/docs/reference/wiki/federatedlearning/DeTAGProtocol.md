---
title: "DeTAGProtocol<T>"
description: "Implements DeTAG — Decentralized gradient Tracking for exact convergence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Decentralized`

Implements DeTAG — Decentralized gradient Tracking for exact convergence.

## For Beginners

Basic decentralized averaging has a problem: it converges to
a "consensus" point that may not be the true global minimum because each client only sees
their local gradient. DeTAG (Decentralized gradient Tracking) fixes this by having each
client track the difference between their local gradient and the global gradient estimate.
This correction term ensures exact convergence even with heterogeneous data.

## How It Works

Gradient tracking update:

Reference: Li, H., et al. (2023). "DeTAG: Decentralized Tracking-based
Asynchronous Gradient methods." 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeTAGProtocol(Double)` | Creates a new DeTAG protocol. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets the learning rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTracker(Int32)` | Gets the current gradient tracker for a client (for sharing with neighbors). |
| `Step(Int32,[],[],[],Dictionary<Int32,[]>,Dictionary<Int32,[]>,Dictionary<Int32,Double>)` | Performs one DeTAG update step for a client, including neighbor-averaged gradient tracking. |


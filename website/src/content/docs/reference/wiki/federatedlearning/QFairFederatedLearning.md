---
title: "QFairFederatedLearning<T>"
description: "Implements q-FFL (q-Fair Federated Learning) — parameterized fairness via power-mean."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Implements q-FFL (q-Fair Federated Learning) — parameterized fairness via power-mean.

## For Beginners

q-FFL provides a tunable knob for fairness. The parameter q
controls how much we care about the worst-off clients: q=0 gives standard FedAvg (optimize
average loss), q=1 weights clients proportionally to their loss, and q→∞ gives minimax
fairness (optimize worst-case). This lets you smoothly trade off average performance for
fairness.

## How It Works

Objective:

Reference: Li, T., et al. (2020). "Fair Resource Allocation in Federated Learning."
ICLR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QFairFederatedLearning(Double)` | Creates a new q-FFL instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Q` | Gets the fairness parameter q. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeObjective(Dictionary<Int32,Double>)` | Computes the q-fair objective value. |
| `ComputeWeights(Dictionary<Int32,Double>)` | Computes q-fair aggregation weights based on client losses. |


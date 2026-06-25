---
title: "MoonAggregationStrategy<T>"
description: "Implements MOON (Model-COntrastive Learning) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements MOON (Model-COntrastive Learning) aggregation strategy.

## For Beginners

MOON corrects "local drift" by adding a contrastive loss
during client training. The contrastive loss pulls the local model's representation closer
to the global model and pushes it away from the previous local model, reducing divergence
caused by non-IID data.

## How It Works

During aggregation, MOON uses standard weighted averaging (same as FedAvg). The key
innovation is in the local training objective, which includes:

The contrastive loss uses cosine similarity scaled by temperature:

Reference: Li, Q., He, B., and Song, D. (2021). "Model-Contrastive Federated Learning."
CVPR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoonAggregationStrategy(Double,Double)` | Initializes a new instance of the `MoonAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContrastiveWeight` | Gets the contrastive loss weight (mu). |
| `Temperature` | Gets the contrastive temperature parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates client models using weighted averaging. |
| `ComputeContrastiveLoss(Vector<>,Vector<>,Vector<>)` | Computes the MOON contrastive loss for a client during local training. |
| `ComputeTotalLoss(,Vector<>,Vector<>,Int32)` | Computes the complete local training loss including both task loss and contrastive loss. |
| `GetPreviousRepresentation(Int32)` | Retrieves the previous round's local representation for a client. |
| `GetStrategyName` |  |
| `StoreRepresentation(Int32,Vector<>)` | Stores the current local representation for a client to be used as the negative pair in the next round. |


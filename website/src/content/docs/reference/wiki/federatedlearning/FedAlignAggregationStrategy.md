---
title: "FedAlignAggregationStrategy<T>"
description: "Implements FedAlign (Feature Alignment) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements FedAlign (Feature Alignment) aggregation strategy.

## For Beginners

Different clients with different data can learn different "languages"
for representing the same concepts. FedAlign adds a regularizer during local training that
forces each client's feature space to align with shared anchor representations, so all clients
speak the same "language" when their models are combined.

## How It Works

Local training objective:

where anchors are shared reference inputs and D measures representation distance
using either L2 distance or CKA (Centered Kernel Alignment).

Protocol:

Reference: Mendieta, M., et al. (2022). "Local Learning Matters: Rethinking Data Heterogeneity
in Federated Learning." CVPR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedAlignAggregationStrategy(Double,AlignmentDistanceMetric)` | Initializes a new instance of the `FedAlignAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignmentWeight` | Gets the feature alignment weight (alpha). |
| `Anchors` | Gets the registered anchor inputs, or null if none have been set. |
| `DistanceMetric` | Gets the distance metric used for alignment. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates client models using standard weighted averaging. |
| `ComputeAlignmentLoss(Matrix<>,Matrix<>)` | Computes the feature alignment loss between local and global model representations on anchors. |
| `ComputeCKADistance(Matrix<>,Matrix<>)` | Centered Kernel Alignment distance: 1 - CKA(local, global). |
| `ComputeL2Distance(Matrix<>,Matrix<>)` | Mean squared L2 distance: (1/N) * sum_i \|\|f_local_i - f_global_i\|\|^2. |
| `ComputeMMDDistance(Matrix<>,Matrix<>)` | Maximum Mean Discrepancy with RBF kernel: MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]. |
| `ComputeTotalLoss(,Matrix<>,Matrix<>)` | Computes the complete local training loss including task loss and alignment loss. |
| `GetStrategyName` |  |
| `SetAnchors(Matrix<>)` | Registers anchor inputs that all clients use for alignment. |


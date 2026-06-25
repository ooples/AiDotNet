---
title: "FederatedOrthogonalProjection<T>"
description: "Federated Orthogonal Projection — prevents forgetting by projecting gradients to be orthogonal to the subspace of previously important parameter directions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ContinualLearning`

Federated Orthogonal Projection — prevents forgetting by projecting gradients to be
orthogonal to the subspace of previously important parameter directions.

## For Beginners

Think of it as building new knowledge on a "wall" that's perpendicular
to the old knowledge. If the old knowledge lives on the floor, new knowledge goes up the wall
— they never conflict. In federated settings, each client reports which directions are important,
and the global projection space is the union of all clients' important directions.

## How It Works

Orthogonal gradient projection (Farajtabar et al., 2020; Zhang et al., ICCV 2025)
computes the subspace spanned by important parameter directions from previous tasks,
then projects current gradients to be orthogonal to this subspace. This guarantees
that new learning does not interfere with previously learned knowledge.

References:
Farajtabar et al. (2020), "Orthogonal Gradient Descent for Continual Learning".
Zhang et al. (2025), "FedAGC: Federated Continual Learning with Asymmetric Gradient Correction" (ICCV 2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedOrthogonalProjection(Double)` | Creates a new federated orthogonal projection strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateImportance(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ComputeImportance(Vector<>,Matrix<>)` |  |
| `ComputeRegularizationPenalty(Vector<>,Vector<>,Vector<>,Double)` |  |
| `ProjectGradient(Vector<>,Vector<>)` |  |


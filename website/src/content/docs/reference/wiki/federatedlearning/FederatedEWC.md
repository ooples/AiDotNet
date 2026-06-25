---
title: "FederatedEWC<T>"
description: "Federated Elastic Weight Consolidation (EWC) — prevents forgetting by penalizing changes to parameters that are important for previously learned tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ContinualLearning`

Federated Elastic Weight Consolidation (EWC) — prevents forgetting by penalizing changes
to parameters that are important for previously learned tasks.

## For Beginners

Some model parameters are critical for recognizing cats, while
others are critical for recognizing dogs. EWC identifies which parameters matter for which
tasks and penalizes changing the important ones. In federated EWC, each client reports
which parameters are important for their data, and the server keeps a global importance map.

## How It Works

EWC (Kirkpatrick et al., 2017) uses the Fisher information matrix to estimate which
parameters are important for each task. In federated EWC, the Fisher information is
computed locally at each client and aggregated at the server to form a global estimate
of parameter importance across all tasks and clients.

Reference: Kirkpatrick et al. (2017), "Overcoming Catastrophic Forgetting in Neural Networks".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedEWC(Int32)` | Creates a new Federated EWC strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateImportance(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ComputeImportance(Vector<>,Matrix<>)` |  |
| `ComputeRegularizationPenalty(Vector<>,Vector<>,Vector<>,Double)` |  |
| `ProjectGradient(Vector<>,Vector<>)` |  |


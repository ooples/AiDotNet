---
title: "FedCPPersonalization<T>"
description: "Implements FedCP (Conditional Policy) personalization with input-dependent routing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements FedCP (Conditional Policy) personalization with input-dependent routing.

## For Beginners

Different clients have different types of data. Rather than
forcing all data through the same model path, FedCP learns a "routing policy" per client
that decides which parts of the model to use for each input. The policy network is lightweight
and personalized (kept local), while the main model modules are shared globally. This way,
each client can effectively use a different "subset" of the global model tailored to their data.

## How It Works

Architecture:

Reference: Zhang, J., et al. (2023). "Federated Learning with Conditional Computation."
KDD 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedCPPersonalization(Int32,Double)` | Creates a new FedCP personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumExperts` | Gets the number of expert modules. |
| `PolicyFraction` | Gets the policy network fraction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineExpertOutputs([][],Double[])` | Combines expert outputs using routing weights: output = sum(w_k * expert_k(input)). |
| `ComputeLoadBalancingLoss(Double[][])` | Computes the load-balancing loss to prevent expert collapse (all traffic to one expert). |
| `ComputeRoutingWeights([])` | Computes routing weights from the policy network output using softmax. |
| `ExtractPolicyParameters(Dictionary<String,[]>)` | Extracts local policy network parameters (not aggregated). |
| `ExtractSharedParameters(Dictionary<String,[]>)` | Extracts shared expert module parameters for aggregation. |


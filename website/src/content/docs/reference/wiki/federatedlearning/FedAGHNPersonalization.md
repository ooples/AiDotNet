---
title: "FedAGHNPersonalization<T>"
description: "Implements FedAGHN (Adaptive Gradient-based Heterogeneous Networks) personalization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements FedAGHN (Adaptive Gradient-based Heterogeneous Networks) personalization.

## For Beginners

In standard FL, all clients must use the same model architecture.
FedAGHN relaxes this: each client can have a differently-sized model (e.g., a phone uses
a small model, a workstation uses a large one). It works by defining a shared "knowledge
representation" space and learning adapter layers that project each client's heterogeneous
model into this shared space for aggregation. Gradient similarity across the shared space
determines aggregation weights adaptively.

## How It Works

Architecture:

Reference: FedAGHN: Adaptive Gradient Heterogeneous Networks for FL (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedAGHNPersonalization(Int32,Double)` | Creates a new FedAGHN personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptiveWeightMomentum` | Gets the adaptive weight momentum. |
| `SharedDimension` | Gets the shared knowledge space dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAdaptiveWeights([],Dictionary<Int32,[]>,Dictionary<Int32,Double>)` | Computes adaptive aggregation weights for a target client based on gradient similarity with all other clients. |
| `ComputeGradientSimilarity([],[])` | Computes gradient similarity between two clients in the shared space using cosine similarity. |
| `ProjectToLocal([],Int32)` | Projects shared-space parameters back to client's local dimension. |
| `ProjectToShared([])` | Projects client parameters from local dimension to shared space for aggregation. |
| `ProjectWithMatrix([],[])` | Projects client parameters using a learned linear projection matrix rather than truncation. |


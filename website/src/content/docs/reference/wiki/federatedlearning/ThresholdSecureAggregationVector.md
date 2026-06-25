---
title: "ThresholdSecureAggregationVector<T>"
description: "Implements dropout-resilient secure aggregation for vector-based model updates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy`

Implements dropout-resilient secure aggregation for vector-based model updates.

## How It Works

**For Beginners:** Secure aggregation lets the server compute the sum/average of client updates
without learning any single client's update. Each client adds random "masks" to its update. When the
server combines all masked updates, the masks cancel out and the server recovers only the aggregate.

This variant is *dropout-resilient*:

- Some clients may fail to upload masked updates (upload dropout).
- Some clients may upload but fail to complete the unmasking step (unmasking dropout).

As long as enough clients complete the unmasking step (the reconstruction threshold), the server can
still recover the aggregate by reconstructing missing self-masks using Shamir secret sharing and by
removing leftover pairwise masks for clients that did not upload.

Reference: Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning."

## Properties

| Property | Summary |
|:-----|:--------|
| `MinimumUploaderCount` | Gets the minimum number of clients that must upload masked updates for the round to succeed. |
| `ReconstructionThreshold` | Gets the reconstruction threshold required to complete unmasking. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeRound(List<Int32>,Int32,Int32,Double)` | Initializes a new secure aggregation round by generating the required cryptographic material. |


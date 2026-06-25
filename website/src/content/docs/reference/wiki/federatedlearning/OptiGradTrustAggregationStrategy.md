---
title: "OptiGradTrustAggregationStrategy<T>"
description: "Implements OptiGradTrust (Optimized Gradient Trust) aggregation strategy with historical reputation tracking."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements OptiGradTrust (Optimized Gradient Trust) aggregation strategy with
historical reputation tracking.

## For Beginners

OptiGradTrust builds on trust-based defenses like FLTrust by
adding a historical reputation system. Each client maintains a trust score that is updated
over multiple rounds. A client that consistently sends aligned, constructive updates builds
a higher reputation, while one that repeatedly deviates gets downweighted. This makes the
defense more resilient to adaptive attackers who behave honestly for a few rounds then
suddenly attack.

## How It Works

Trust update rule:

Reference: Optimized Gradient Trust Scoring for Federated Learning (2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OptiGradTrustAggregationStrategy(Double,Double)` | Initializes a new instance of the `OptiGradTrustAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MinReputation` | Gets the minimum reputation floor. |
| `Momentum` | Gets the EMA momentum for reputation updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `GetStrategyName` |  |


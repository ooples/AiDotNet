---
title: "VerifiableAggregationStrategy<TModel>"
description: "Decorator that wraps any `IAggregationStrategy` with proof verification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Decorator that wraps any `IAggregationStrategy` with proof verification.

## For Beginners

This class wraps an existing aggregation strategy (like FedAvg or Krum)
and adds verification: before aggregating updates, it checks cryptographic proofs from each
client. Only clients that pass verification are included in the aggregation.

## How It Works

**How to use:**

**Integration with existing FL:** The existing Byzantine-robust aggregators (Krum, Bulyan,
Median) detect statistical anomalies. This adds cryptographic guarantees on top — first
verify proofs, then pass verified updates to the inner aggregator.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VerifiableAggregationStrategy(IAggregationStrategy<>,VerificationOptions)` | Initializes a new instance of `VerifiableAggregationStrategy`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the verification options. |
| `RejectedClients` | Gets the list of client IDs rejected in the current round. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,>,Dictionary<Int32,Double>)` |  |
| `GetStrategyName` |  |
| `RegisterClientProof(Int32,ClientProofBundle)` | Registers a client's proof bundle for verification during aggregation. |
| `SetRound(Int32)` | Sets the current training round (for proof verification context). |


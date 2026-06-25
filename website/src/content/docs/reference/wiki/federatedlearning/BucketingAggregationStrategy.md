---
title: "BucketingAggregationStrategy<T>"
description: "Implements the Bucketing meta-strategy for Byzantine-robust federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements the Bucketing meta-strategy for Byzantine-robust federated learning.

## For Beginners

Bucketing is not an aggregation rule itself — it is a
"wrapper" that improves any existing robust aggregation rule. Before running the inner
aggregator, clients are randomly shuffled into equal-sized buckets. Within each bucket
the client updates are averaged, producing one "super-update" per bucket. The inner robust
aggregator then operates on these super-updates instead of individual client updates.

## How It Works

This provably increases the *breakdown point* (the fraction of adversaries
the defense can tolerate) for any sub-quadratic robust aggregator.

Algorithm:

Reference: Karimireddy, S. P., et al. (2022). "Byzantine-Robust Learning on
Heterogeneous Datasets via Bucketing." ICML 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BucketingAggregationStrategy(ParameterDictionaryAggregationStrategyBase<>,Int32,Int32)` | Initializes a new instance of the `BucketingAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InnerStrategy` | Gets the inner robust aggregation strategy. |
| `NumBuckets` | Gets the number of buckets. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `GetStrategyName` |  |


---
title: "BloomFilterPsi"
description: "Implements Bloom filter based probabilistic Private Set Intersection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements Bloom filter based probabilistic Private Set Intersection.

## For Beginners

A Bloom filter is like a very compact checklist that can answer
"is this item on the list?" with two possible answers:

## How It Works

Bloom filter PSI uses a probabilistic data structure to represent one party's set.
The other party queries the Bloom filter for each of its elements. Matches indicate
probable intersection membership, with a configurable false-positive rate.

The false-positive rate is configurable. A rate of 0.001 means roughly 1 in 1000
non-matching elements may be incorrectly reported as matching. This is usually acceptable
when followed by a verification step.

**Complexity:** O(n+m) time, O(n) space for the Bloom filter (much smaller than
storing the full set). Fastest PSI protocol but probabilistic.

**Security note:** Standard Bloom filter PSI leaks membership information.
For privacy, the Bloom filter should be encrypted or transmitted via secure channel.
This implementation simulates the protocol assuming a secure channel.

**Reference:** Dong et al., "When Private Set Intersection Meets Big Data",
ACM CCS 2013.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeExactCardinality(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `ComputeOptimalFilterSize(Int32,Double)` | Computes optimal Bloom filter size: m = -n*ln(p) / (ln(2))^2 |
| `ComputeOptimalHashCount(Int32,Int32)` | Computes optimal number of hash functions: k = (m/n) * ln(2) |
| `InsertIntoFilter(Boolean[],String,Int32[])` | Inserts an element into the Bloom filter by setting the bits at each hash position. |
| `QueryFilter(Boolean[],String,Int32[])` | Queries the Bloom filter for membership. |


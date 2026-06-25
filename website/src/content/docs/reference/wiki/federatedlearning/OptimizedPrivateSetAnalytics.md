---
title: "OptimizedPrivateSetAnalytics<T>"
description: "Implements Optimized Private Set Analytics (OPSA) beyond basic intersection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements Optimized Private Set Analytics (OPSA) beyond basic intersection.

## For Beginners

Standard PSI (Private Set Intersection) tells you which items
two parties share in common. OPSA extends this to richer analytics: set union cardinality
(how many unique items total?), frequency estimation (how common is each item?), and threshold
queries (which items appear in at least k parties?). All operations are private — no party
learns the other parties' raw sets.

## How It Works

Supported operations:

Reference: Optimized Private Set Analytics for Federated Learning (2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OptimizedPrivateSetAnalytics(Int32,Int32,Int32,Int32)` | Creates a new OPSA instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HLLPrecision` | Gets the HyperLogLog precision parameter (p). |
| `HLLRegisterCount` | Gets the number of HyperLogLog registers (2^p). |
| `SketchDepth` | Gets the sketch depth. |
| `SketchWidth` | Gets the sketch width. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateHLLRegisters(IEnumerable<String>)` | Creates a HyperLogLog register array from a set of items. |
| `CreateSecretShares(Int32[0:,0:],Int32)` | Creates additive secret shares of a count-min sketch for threshold queries. |
| `CreateSketch(IEnumerable<String>)` | Creates a count-min sketch from a set of items. |
| `EstimateCardinality(Byte[])` | Estimates the cardinality (number of distinct elements) from HLL registers. |
| `EstimateFrequency(Int32[0:,0:],String)` | Estimates the frequency of an item from merged sketches. |
| `EstimateIntersectionCardinality(Byte[],Byte[])` | Estimates intersection cardinality of two parties using inclusion-exclusion: \|A ∩ B\| = \|A\| + \|B\| - \|A ∪ B\|. |
| `EstimateJaccardSimilarity(Byte[],Byte[])` | Estimates the Jaccard similarity between two parties: \|A ∩ B\| / \|A ∪ B\|. |
| `EstimateUnionCardinality(IReadOnlyList<Byte[]>)` | Estimates union cardinality by merging HLL registers from multiple parties. |
| `MergeHLLRegisters(IReadOnlyList<Byte[]>)` | Merges HLL registers from multiple parties (element-wise max). |
| `MergeSketches(IReadOnlyList<Int32[0:,0:]>)` | Merges sketches from multiple parties (element-wise sum). |
| `ThresholdQuery(Int32[0:,0:],IEnumerable<String>,Int32)` | Finds items from a candidate set that appear at least `threshold` times across all parties, using merged count-min sketches. |


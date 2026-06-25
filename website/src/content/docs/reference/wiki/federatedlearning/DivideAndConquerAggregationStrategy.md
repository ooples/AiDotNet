---
title: "DivideAndConquerAggregationStrategy<T>"
description: "Implements DnC (Divide and Conquer) aggregation strategy for Byzantine-robust FL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements DnC (Divide and Conquer) aggregation strategy for Byzantine-robust FL.

## For Beginners

Some poisoning attacks are hard to detect when you look at the
full high-dimensional update vectors — the malicious signal hides in the noise.
DnC projects client updates into random low-dimensional subspaces, then uses spectral
analysis (top singular vector projection) to identify attackers that might evade
simpler coordinate-wise defenses like median or trimmed mean.

## How It Works

Algorithm:

Reference: Shejwalkar, V. & Houmansadr, A. (2021). "Manipulating the Byzantine:
Optimizing Model Poisoning Attacks and Defenses for Federated Learning."
NDSS 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DivideAndConquerAggregationStrategy(Int32,Int32,Int32)` | Initializes a new instance of the `DivideAndConquerAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumByzantine` | Gets the expected number of Byzantine clients. |
| `SubspaceDimension` | Gets the random projection subspace dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `ComputeTopRightSingularVector(Double[][],Int32,Int32,Random)` | Computes the top right singular vector of the data matrix via power iteration. |
| `GenerateOrthogonalProjection(Random,Int32,Int32)` | Generates an orthogonal projection matrix via modified Gram-Schmidt on Gaussian vectors. |
| `GetStrategyName` |  |


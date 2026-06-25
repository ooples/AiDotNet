---
title: "VariationOfInformation<T>"
description: "Variation of Information (VI) for comparing clustering results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Variation of Information (VI) for comparing clustering results.

## For Beginners

Variation of Information asks "How different are these groupings?"

It measures the information distance between two clusterings:

- VI = 0: Identical clusterings (no information lost or gained)
- VI > 0: Different clusterings (some information differs)

Unlike other metrics, LOWER is better for VI (it's a distance measure).

Think of it as: "How much would I need to change one grouping to get the other?"

## How It Works

Variation of Information is an information-theoretic measure of the distance
between two clusterings. It measures the amount of information lost and gained
when going from one clustering to another.

VI(C, K) = H(C) + H(K) - 2*I(C, K)
= H(C|K) + H(K|C)
Where:

- H(C) = entropy of true clustering
- H(K) = entropy of predicted clustering
- I(C, K) = mutual information
- H(C|K) = conditional entropy of C given K

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationOfInformation(Boolean)` | Initializes a new VariationOfInformation instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |
| `ComputeAMI(Vector<>,Vector<>)` | Computes the Adjusted Mutual Information (AMI). |
| `ComputeExpectedMI(Dictionary<Int32,Int32>,Dictionary<Int32,Int32>,Int32)` | Computes the expected mutual information under the hypergeometric null model. |
| `ComputeNMI(Vector<>,Vector<>)` | Computes the Normalized Mutual Information (NMI). |


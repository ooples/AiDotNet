---
title: "NormalizedMutualInformation<T>"
description: "Normalized Mutual Information for comparing cluster assignments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Normalized Mutual Information for comparing cluster assignments.

## For Beginners

NMI measures shared information between clusterings.

Mutual Information asks:
"How much does knowing clustering U tell me about clustering V?"

Entropy H(U) measures uncertainty:

- Low entropy: Few clusters, predictable
- High entropy: Many equal-sized clusters, less predictable

NMI normalizes by average entropy so:

- NMI = 1: Knowing U completely determines V
- NMI = 0: U tells nothing about V (independent)

Unlike ARI, NMI is always non-negative.

## How It Works

Normalized Mutual Information (NMI) measures the mutual information
between two clusterings, normalized to range [0, 1]:

- 1: Perfect agreement
- 0: No mutual information (independent)

NMI = 2 * I(U, V) / (H(U) + H(V))
Where:

- I(U, V) = Mutual information between clusterings U and V
- H(U), H(V) = Entropy of each clustering

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NormalizedMutualInformation(NMINormalization)` | Initializes a new NormalizedMutualInformation instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |


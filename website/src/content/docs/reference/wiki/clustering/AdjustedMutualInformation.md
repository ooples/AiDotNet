---
title: "AdjustedMutualInformation<T>"
description: "Adjusted Mutual Information (AMI) for comparing clustering results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Adjusted Mutual Information (AMI) for comparing clustering results.

## For Beginners

AMI is like NMI but corrected for luck.

Regular NMI can be high just by chance if you have many clusters.
AMI asks "How much better than random?"

- AMI = 0: No better than random
- AMI = 1: Perfect agreement
- AMI can be negative if worse than random

## How It Works

AMI adjusts NMI for chance agreement. Random clusterings will have
AMI close to 0, while perfect agreement gives AMI = 1.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdjustedMutualInformation` | Initializes a new AdjustedMutualInformation instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Vector<>,Vector<>)` |  |


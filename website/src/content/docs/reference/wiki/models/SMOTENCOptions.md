---
title: "SMOTENCOptions<T>"
description: "Configuration options for SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features), a k-NN based oversampling method that generates synthetic minority samples by interpolating between existing ones."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for SMOTE-NC (Synthetic Minority Over-sampling Technique for
Nominal and Continuous features), a k-NN based oversampling method that generates
synthetic minority samples by interpolating between existing ones.

## For Beginners

SMOTE-NC helps when your data is imbalanced
(e.g., 99% normal transactions, 1% fraud).

How it works:

1. Find each minority sample's k nearest neighbors (similar samples)
2. Pick one neighbor randomly
3. Create a new sample "in between" the original and the neighbor
- For numbers: pick a random point on the line between them
- For categories: use the most common category among the neighbors

This produces new minority samples that are realistic because they're
based on actual data relationships, not just random copies.

Example:

## How It Works

SMOTE-NC extends the original SMOTE algorithm to handle mixed-type data:

- **Continuous features**: Linear interpolation between a sample and its k-nearest neighbor
- **Categorical features**: Majority vote among the k-nearest neighbors
- **Distance metric**: Euclidean for continuous + Value Difference Metric for categoricals

Reference: "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets or sets the number of nearest neighbors to consider. |
| `LabelColumnIndex` | Gets or sets the index of the label column for identifying minority class. |
| `MinorityClassValue` | Gets or sets the minority class value to oversample. |


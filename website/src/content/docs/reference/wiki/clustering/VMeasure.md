---
title: "VMeasure<T>"
description: "Computes the V-Measure for cluster-label agreement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Computes the V-Measure for cluster-label agreement.

## For Beginners

V-Measure combines two important properties:

Homogeneity: "Is each cluster pure?"

- A cluster is pure if it contains only one type of item
- Example: A "cats" cluster should have only cats

Completeness: "Is each class together?"

- All items of the same type should be in the same cluster
- Example: All cats should be in the same cluster

V-Measure balances these:

- High homogeneity + low completeness = many small pure clusters
- Low homogeneity + high completeness = one big impure cluster
- High V-Measure = both are high (ideal)

V-Measure ranges from 0 to 1, where 1 is perfect agreement.

## How It Works

V-Measure is the harmonic mean of homogeneity and completeness, providing
a single metric that balances both properties. It requires ground truth labels.

Components:

- Homogeneity: Each cluster contains only members of a single class
- Completeness: All members of a given class are assigned to the same cluster
- V-Measure: 2 * (homogeneity * completeness) / (homogeneity + completeness)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VMeasure(Double)` | Initializes a new VMeasure instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HigherIsBetter` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Clustering#Evaluation#IExternalClusterMetric{T}#Compute(Vector<>,Vector<>)` |  |
| `Compute(Matrix<>,Vector<>)` |  |
| `ComputeCompleteness(Vector<>,Vector<>)` | Computes completeness score. |
| `ComputeHomogeneity(Vector<>,Vector<>)` | Computes homogeneity score. |
| `ComputeWithTrueLabels(Matrix<>,Vector<>,Vector<>)` | Computes V-Measure comparing predicted labels to true labels. |


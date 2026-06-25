---
title: "BorderlineSMOTE<T>"
description: "Implements Borderline-SMOTE for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements Borderline-SMOTE for handling imbalanced datasets.

## For Beginners

Regular SMOTE creates synthetic samples uniformly among all
minority samples. Borderline-SMOTE is smarter - it identifies which minority samples
are "borderline" (near majority class samples) and only creates synthetics from those.

How it works:

1. For each minority sample, count its majority neighbors
2. Classify each minority sample as:
- SAFE: Mostly minority neighbors (not borderline, skip)
- DANGER: Mix of minority and majority neighbors (borderline, use for synthesis)
- NOISE: Mostly majority neighbors (outlier, skip)
3. Only generate synthetic samples from DANGER samples

Why this is better:

- Creates samples where they're needed most (at the boundary)
- Avoids wasting effort on samples deep in minority territory
- Reduces noise by not synthesizing from outliers

References:

- Han et al. (2005). "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning"

## How It Works

Borderline-SMOTE focuses on minority samples that are near the decision boundary
(borderline samples), as these are the most informative for classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BorderlineSMOTE(Double,Int32,Int32,BorderlineSMOTEKind,Nullable<Int32>)` | Initializes a new instance of the BorderlineSMOTE class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MNeighbors` | Number of neighbors used for borderline detection. |
| `Name` | Gets the name of this oversampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateRegularSMOTE(Matrix<>,List<Int32>,Int32)` | Fallback to regular SMOTE when no borderline samples are found. |
| `GenerateSyntheticSamples(Matrix<>,List<Int32>,Int32)` | Generates synthetic samples using Borderline-SMOTE. |
| `InterpolateSamples(Vector<>,Vector<>)` | Creates a synthetic sample by interpolating between two samples. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_kind` | The variant of Borderline-SMOTE to use. |


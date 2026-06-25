---
title: "SMOTETomek<T>"
description: "Implements SMOTE combined with Tomek Links (SMOTETomek) for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements SMOTE combined with Tomek Links (SMOTETomek) for handling imbalanced datasets.

## For Beginners

This is a two-step process:

Step 1 (SMOTE): Create synthetic minority samples

- Increases minority class size
- Fills in the minority class region

Step 2 (Tomek Links): Remove specific boundary pairs

- Only removes samples that form Tomek links
- More targeted than ENN
- Preserves more data

Comparison with SMOTEENN:

- SMOTETomek: Removes fewer samples, more conservative
- SMOTEENN: Removes more samples, more aggressive cleaning

Visual example:
```
Original: M M M M M M M M M m m
After SMOTE: M M M M M M M M M m m m m m m m
^ ^
Tomek link pair
After Tomek: M M M M M M M M m m m m m m m
^
Only the majority sample of the pair removed
```

When to use:

- When you want to balance but preserve more data
- When ENN is too aggressive for your dataset
- As a middle ground between pure SMOTE and SMOTEENN

References:

- Batista et al. (2004). "A Study of the Behavior of Several Methods for Balancing

Machine Learning Training Data"

## How It Works

SMOTETomek first applies SMOTE to oversample the minority class, then applies Tomek Links
removal to clean up the decision boundary. This is less aggressive than SMOTEENN.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SMOTETomek(Double,Int32,Nullable<Int32>)` | Initializes a new instance of the SMOTETomek class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this resampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetStatistics` | Gets statistics about the last resampling operation. |
| `Resample(Matrix<>,Vector<>)` | Resamples the dataset using SMOTE followed by Tomek Links removal. |


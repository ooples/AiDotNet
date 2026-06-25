---
title: "SMOTEENN<T>"
description: "Implements SMOTE combined with Edited Nearest Neighbors (SMOTEENN) for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements SMOTE combined with Edited Nearest Neighbors (SMOTEENN) for handling imbalanced datasets.

## For Beginners

This is a two-step process:

Step 1 (SMOTE): Create synthetic minority samples

- Increases minority class size
- Fills in the minority class region

Step 2 (ENN): Clean up the boundary

- Removes samples misclassified by their neighbors
- Cleans both majority AND synthetic minority samples
- Creates a cleaner decision boundary

Why combine them:

- SMOTE alone can create noisy synthetic samples
- ENN cleans up these noisy samples
- Result is a balanced AND clean dataset

Visual example:
```
Original: M M M M M M M M M m m
After SMOTE: M M M M M M M M M m m m m m m m
^ synthetic
After ENN: M M M M M M m m m m m m
^ ^
Noisy samples removed from both classes
```

References:

- Batista et al. (2004). "A Study of the Behavior of Several Methods for Balancing

Machine Learning Training Data"

## How It Works

SMOTEENN first applies SMOTE to oversample the minority class, then applies ENN to
remove noisy and borderline samples from both classes. This combination often produces
better results than either method alone.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SMOTEENN(Double,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the SMOTEENN class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this resampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetStatistics` | Gets statistics about the last resampling operation. |
| `Resample(Matrix<>,Vector<>)` | Resamples the dataset using SMOTE followed by ENN. |


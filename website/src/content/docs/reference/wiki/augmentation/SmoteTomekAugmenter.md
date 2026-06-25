---
title: "SmoteTomekAugmenter<T>"
description: "Implements SMOTE-Tomek combination for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Implements SMOTE-Tomek combination for imbalanced datasets.

## For Beginners

SMOTE-Tomek combines SMOTE oversampling with Tomek links cleaning.
First, SMOTE generates synthetic minority samples, then Tomek links are removed to clean
the class boundary and reduce noise.

## How It Works

**How it works:**

- Apply SMOTE to generate synthetic minority samples
- Identify Tomek links (pairs of samples from different classes that are mutual nearest neighbors)
- Remove the majority class samples from Tomek links

**Benefits:**

- Balances classes while cleaning the decision boundary
- Reduces noise introduced by SMOTE near the boundary
- More robust than SMOTE alone

**Reference:** Batista et al., "A Study of the Behavior of Several Methods for Balancing
Machine Learning Training Data" (2004)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SmoteTomekAugmenter(Int32,Double,Double)` | Creates a new SMOTE-Tomek augmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KNeighbors` | Gets the number of nearest neighbors used by SMOTE. |
| `SamplingRatio` | Gets the SMOTE sampling ratio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `ApplySmoteTomek(Matrix<>,Vector<>,AugmentationContext<>)` | Applies SMOTE-Tomek to a labeled dataset. |
| `GetParameters` |  |


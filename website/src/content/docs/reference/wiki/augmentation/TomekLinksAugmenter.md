---
title: "TomekLinksAugmenter<T>"
description: "Implements Tomek Links removal for cleaning decision boundaries in imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Implements Tomek Links removal for cleaning decision boundaries in imbalanced datasets.

## For Beginners

A Tomek link is a pair of samples from different classes that
are each other's nearest neighbor. These pairs are typically noisy or borderline samples.
Removing Tomek links cleans the decision boundary, making classification easier.

## How It Works

**What is a Tomek Link:**
Two samples (a, b) form a Tomek link if:

- a and b belong to different classes
- a is b's nearest neighbor
- b is a's nearest neighbor

These samples are ambiguous because they're very close to samples of the opposite class.

**Removal Strategies:**

- RemoveBoth: Remove both samples in the Tomek link (aggressive cleaning)
- RemoveMajority: Only remove the majority class sample (preserves minority)
- RemoveMinority: Only remove the minority class sample (rarely used)

**When to use:**

- After SMOTE to clean noisy synthetic samples (SMOTE-Tomek)
- Before training to clean overlapping regions
- When precision is more important than recall

**Reference:** Tomek, "Two Modifications of CNN" (1976)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TomekLinksAugmenter(TomekLinksAugmenter<>.RemovalStrategy,Double)` | Creates a new Tomek Links augmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Strategy` | Gets the removal strategy to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `CountTomekLinks(Matrix<>,Vector<>,)` | Gets the number of Tomek links in the dataset. |
| `FindNearestNeighbor(Matrix<>,Int32)` | Finds the nearest neighbor for a given sample. |
| `FindTomekLinkIndices(Matrix<>,Vector<>,)` | Finds indices of samples that form Tomek links. |
| `GetParameters` |  |
| `RemoveTomekLinks(Matrix<>,Vector<>,)` | Removes samples that form Tomek links between classes. |


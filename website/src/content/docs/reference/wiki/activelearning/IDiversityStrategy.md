---
title: "IDiversityStrategy<T, TInput, TOutput>"
description: "Interface for diversity-based sampling strategies in active learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for diversity-based sampling strategies in active learning.

## For Beginners

Diversity strategies select samples that are diverse or
representative of the data distribution. This helps ensure the model learns from
a variety of examples rather than similar ones.

## How It Works

**Common Diversity Strategies:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of the diversity strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistanceMatrix(IDataset<,,>)` | Computes pairwise distances between samples. |
| `ComputeDiversityScores(IDataset<,,>,IDataset<,,>)` | Computes diversity scores for samples in the unlabeled pool. |
| `GetFeatureRepresentation()` | Gets the feature representation for a sample. |
| `SelectDiverseSamples(IDataset<,,>,IDataset<,,>,Int32)` | Selects diverse samples from the unlabeled pool. |


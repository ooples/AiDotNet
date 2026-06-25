---
title: "SMOTENCGenerator<T>"
description: "SMOTE-NC generator that creates synthetic minority samples by interpolating between existing minority samples and their k-nearest neighbors, supporting both continuous and categorical features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

SMOTE-NC generator that creates synthetic minority samples by interpolating between
existing minority samples and their k-nearest neighbors, supporting both continuous
and categorical features.

## For Beginners

SMOTE-NC generates new minority samples by "mixing" existing ones:

Given a minority sample and a nearby neighbor:

- For numbers (age, income): pick a random point between the two values

Example: if sample has age=30 and neighbor has age=40, synthetic might have age=35

- For categories (gender, region): use the most common value among the k neighbors

Example: if 3 of 5 neighbors have region="East", synthetic gets region="East"

This is simpler and faster than GANs, and works well for structured tabular data.

## How It Works

SMOTE-NC (Nominal and Continuous) operates as follows:

1. Extract all minority class samples from the training data
2. For each minority sample, find its k nearest neighbors among other minority samples
3. Generate synthetic samples by interpolating between samples and randomly chosen neighbors

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SMOTENCGenerator(SMOTENCOptions<>)` | Initializes a new instance of the `SMOTENCGenerator` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistance(Int32,Int32)` | Computes the mixed-type distance between two samples using the SMOTE-NC formula. |
| `ComputeMedianStdOfContinuousFeatures` | Computes the median standard deviation of continuous features from the minority samples. |
| `FindKNearestNeighbors(Int32,Int32)` | Finds the k nearest neighbors for a given minority sample using mixed-type distance. |
| `FitInternal(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `GenerateInternal(Int32,Vector<>,Vector<>)` |  |
| `MajorityVoteAmongNeighbors(Int32,Int32[])` | Returns the majority vote value for a categorical feature among the given neighbors. |


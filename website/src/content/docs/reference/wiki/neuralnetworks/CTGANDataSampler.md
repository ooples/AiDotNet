---
title: "CTGANDataSampler<T>"
description: "Handles conditional vector generation and training-by-sampling for CTGAN."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Handles conditional vector generation and training-by-sampling for CTGAN.
Implements the sampling strategy from the CTGAN paper that ensures all categories
are represented equally during training.

## For Beginners

Real data often has imbalanced categories. For example, in a
"car color" column, 80% might be "white" and only 2% "yellow". Without special handling,
the generator would mostly learn to produce white cars.

The data sampler fixes this by:

1. Picking a random categorical column
2. Picking a random category (so "yellow" gets equal chance as "white")
3. Finding a real row with that category to use as training data
4. Creating a "conditional vector" that tells the generator what category to produce

This ensures the generator learns to produce all categories well.

## How It Works

The CTGAN paper introduces "training-by-sampling" to handle imbalanced categorical columns:

1. Randomly pick a discrete/categorical column
2. Randomly pick a category value from that column (with equal probability)
3. Construct a conditional vector indicating the selected category
4. Sample a real row that has the selected category value

This ensures that rare categories are sampled proportionally during training,
preventing the generator from ignoring minority classes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CTGANDataSampler(Random)` | Initializes a new `CTGANDataSampler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConditionalVectorWidth` | Gets the width of the conditional vector. |
| `NumDiscreteColumns` | Gets the number of discrete/categorical columns used for conditioning. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateConditionVector(Int32,Int32)` | Generates a conditional vector for a specific category in generation mode. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>)` | Builds the category-to-row-index tables from the training data. |
| `SampleCategoryByLogFreq(CTGANDataSampler<>.DiscreteColumnInfo)` | Samples a category index from a discrete column's log-frequency distribution (Xu 2019 §4.3), falling back to a populated category if the draw lands on an empty one. |
| `SampleCondVecWithMask` | Full training-by-sampling draw (Xu 2019 §4.3) exposing everything the generator's conditional cross-entropy loss needs: the one-hot conditional vector, a MASK that is 1 only over the selected column's category block, the selected discrete-c… |
| `SampleConditionAndRow` | Samples a conditional vector and a corresponding real row index for training. |
| `SampleRandomConditionVector` | Generates a random conditional vector for unconditional generation (picks random category). |


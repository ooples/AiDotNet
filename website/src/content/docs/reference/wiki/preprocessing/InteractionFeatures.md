---
title: "InteractionFeatures<T>"
description: "Generates pairwise interaction features between input features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureGeneration`

Generates pairwise interaction features between input features.

## For Beginners

Interaction features capture combined effects:

- If both "age" and "income" matter together (not just separately)
- Creating age × income might help the model
- This is simpler than full polynomial features (no squared terms)

## How It Works

InteractionFeatures creates new features by multiplying pairs of existing features.
Unlike PolynomialFeatures with degree=2, this only produces interaction terms,
not squared terms.

For features [a, b, c], this produces: [ab, ac, bc]

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InteractionFeatures(Boolean,InteractionType,Int32[])` | Creates a new instance of `InteractionFeatures`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IncludeOriginal` | Gets whether original features are included in output. |
| `InteractionPairs` | Gets the interaction pairs (feature indices). |
| `InteractionType` | Gets the type of interactions generated. |
| `NOutputFeatures` | Gets the number of output features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the transformer by computing interaction pairs. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms data by generating interaction features. |


---
title: "PolynomialFeatures<T>"
description: "Generates polynomial and interaction features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureGeneration`

Generates polynomial and interaction features.

## For Beginners

This transformer creates new features from existing ones:

- Polynomial terms: a, a², a³, etc.
- Interaction terms: a*b, a*b*c, etc.
- Useful for capturing non-linear relationships

Example with degree=2 and features [x₁, x₂]:
Output: [1, x₁, x₂, x₁², x₁x₂, x₂²]

## How It Works

PolynomialFeatures generates a new feature matrix consisting of all polynomial combinations
of the features with degree less than or equal to the specified degree. For example, with
degree=2 and input [a, b], generates [1, a, b, a², ab, b²].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolynomialFeatures(Int32,Boolean,Boolean,Int32[])` | Creates a new instance of `PolynomialFeatures`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Degree` | Gets the degree of polynomial features. |
| `IncludeBias` | Gets whether a bias (constant) column is included. |
| `InteractionOnly` | Gets whether only interaction features are generated. |
| `NInputFeatures` | Gets the number of input features. |
| `NOutputFeatures` | Gets the number of output features after transformation. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Learns the polynomial feature combinations from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for polynomial features. |
| `TransformCore(Matrix<>)` | Transforms the data by generating polynomial features. |


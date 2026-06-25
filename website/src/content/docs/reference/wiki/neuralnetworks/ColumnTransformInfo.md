---
title: "ColumnTransformInfo"
description: "Describes how a single original column maps into the transformed representation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Describes how a single original column maps into the transformed representation.

## For Beginners

When the transformer processes your data, each column gets
expanded into a different number of features:

- A continuous column becomes 1 (normalized value) + K (mode indicators) features
- A categorical column becomes N (one-hot categories) features

This info tracks where each original column's features start and how wide they are.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColumnTransformInfo(Boolean,Int32,Int32,Int32)` | Initializes a new `ColumnTransformInfo`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Index` | Index into the continuous or categorical info arrays. |
| `IsContinuous` | Whether this column is continuous (true) or categorical (false). |
| `StartOffset` | Starting offset in the transformed data vector. |
| `Width` | Number of features this column occupies in the transformed representation. |


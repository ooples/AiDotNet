---
title: "HashingEncoder<T>"
description: "Encodes categorical features using feature hashing (hashing trick)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Encoders`

Encodes categorical features using feature hashing (hashing trick).

## For Beginners

Instead of creating one column per category:

- Hash encoding creates a fixed number of columns (e.g., 8)
- Each category is hashed to one of these columns
- Multiple categories may share the same column (collision)

Pros: Fixed memory, handles new categories, fast
Cons: Information loss from collisions, not reversible

## How It Works

HashingEncoder uses a hash function to map categories to a fixed number of columns.
This is useful for high-cardinality categorical features where one-hot encoding
would create too many columns.

Unlike other encoders, HashingEncoder doesn't need to store the category mappings,
making it memory-efficient and able to handle previously unseen categories.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HashingEncoder(Int32,Boolean,Int32[])` | Creates a new instance of `HashingEncoder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlternateSign` | Gets whether alternate signs are used for hash collisions. |
| `NComponents` | Gets the number of hash components (output features per encoded column). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the output feature structure. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for hash encoding. |
| `TransformCore(Matrix<>)` | Transforms the data using feature hashing. |


---
title: "MatrixLayout"
description: "Specifies how data is organized in matrices when working with arrays of data."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies how data is organized in matrices when working with arrays of data.

## For Beginners

A matrix is simply a rectangular grid of numbers arranged in rows and columns.
When working with data in programming, we need to specify how our data is organized.

Think of a spreadsheet:

- You can organize your data by putting similar features in columns (ColumnArrays)
- Or you can organize your data by putting each data point across a row (RowArrays)

For example, if you have data about people (height, weight, age):

ColumnArrays would look like:

- First array: [height1, height2, height3, ...]
- Second array: [weight1, weight2, weight3, ...]
- Third array: [age1, age2, age3, ...]

RowArrays would look like:

- First array: [height1, weight1, age1]
- Second array: [height2, weight2, age2]
- Third array: [height3, weight3, age3]

The choice of layout affects how you access and process your data, and different
algorithms may expect data in different layouts.

## Fields

| Field | Summary |
|:-----|:--------|
| `ColumnArrays` | Data is organized by columns, where each array represents a feature or variable. |
| `RowArrays` | Data is organized by rows, where each array represents a single data point with multiple features. |


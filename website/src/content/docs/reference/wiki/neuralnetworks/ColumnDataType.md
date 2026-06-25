---
title: "ColumnDataType"
description: "Specifies the data type of a column in a tabular dataset."
section: "API Reference"
---

`Enums` · `AiDotNet.NeuralNetworks.SyntheticData`

Specifies the data type of a column in a tabular dataset.

## For Beginners

Each column in your data falls into one of these categories:

- **Continuous**: Numbers that can take any value (e.g., price = 19.99, temperature = 72.5)
- **Discrete**: Integer counts or ordinal numbers (e.g., number of children = 3, rating = 4)
- **Categorical**: Labels or categories (e.g., color = "red", city = "NYC")

The generator uses this information to apply the correct preprocessing and generation strategy.

## Fields

| Field | Summary |
|:-----|:--------|
| `Categorical` | A categorical column (unordered labels, e.g., "red"/"blue"/"green"). |
| `Continuous` | A continuous numerical column (real-valued, e.g., price, temperature). |
| `Discrete` | A discrete integer column (counts or ordinal values, e.g., age in years, number of items). |


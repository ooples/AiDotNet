---
title: "DataComplexity"
description: "Represents the level of complexity in a dataset, which helps determine appropriate model selection and preprocessing."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents the level of complexity in a dataset, which helps determine appropriate model selection and preprocessing.

## For Beginners

Data complexity refers to how difficult it is for a machine learning model to find patterns in your data.

Think of it like solving puzzles:

- Simple data is like a basic jigsaw puzzle with few, large pieces
- Complex data is like an advanced puzzle with many tiny pieces and subtle patterns

Understanding your data's complexity helps you choose the right model:

- Simple data often works well with basic models
- Complex data usually requires more sophisticated models

This enum helps you categorize your data to make better decisions about which algorithms to use.

## Fields

| Field | Summary |
|:-----|:--------|
| `Complex` | Indicates data with intricate patterns, many features, significant noise, or complex dependencies. |
| `Moderate` | Indicates data with somewhat complex patterns, a moderate number of features, and some noise. |
| `Simple` | Indicates data with clear patterns, few features, and minimal noise. |


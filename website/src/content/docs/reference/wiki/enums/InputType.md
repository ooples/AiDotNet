---
title: "InputType"
description: "Specifies the dimensionality of input data for machine learning models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the dimensionality of input data for machine learning models.

## For Beginners

Dimensionality refers to how many separate values are used to represent each data point.
Think of dimensions like coordinates - a 1D point needs just one number (like a position on a line),
a 2D point needs two numbers (like a position on a map), and a 3D point needs three numbers
(like a position in a room).

## Fields

| Field | Summary |
|:-----|:--------|
| `FourDimensional` | Represents four-dimensional input data: [frames, channels, height, width]. |
| `OneDimensional` | Represents input data with a single value per data point. |
| `ThreeDimensional` | Represents input data with three values per data point. |
| `TwoDimensional` | Represents input data with two values per data point. |


---
title: "ToTensorTransform<T>"
description: "Converts a flat array to a Tensor with the specified shape."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Transforms.Numeric`

Converts a flat array to a Tensor with the specified shape.

## For Beginners

When you load image data as a flat array of pixels,
you need to reshape it into the correct dimensions (height x width x channels)
for neural network input.

## How It Works

This transform reshapes a 1D array of values into a multi-dimensional tensor.
The total number of elements must match the product of the shape dimensions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToTensorTransform(Int32[])` | Creates a transform that converts arrays to tensors with the specified shape. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply([])` |  |


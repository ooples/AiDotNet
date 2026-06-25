---
title: "SamplingType"
description: "Specifies the method used to sample or combine values when reducing data dimensions."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the method used to sample or combine values when reducing data dimensions.

## How It Works

**For Beginners:** Sampling is how we summarize a group of numbers into a single value.

In AI, we often need to take a collection of values (like a grid of pixels in an image)
and represent them with fewer values. This process is called "downsampling" or "pooling".

Think of it like summarizing a neighborhood on a map:

- You could pick the tallest building (Max)
- You could calculate the average building height (Average)
- You could use a special mathematical formula (L2Norm)

Different sampling types give different results and are useful in different situations.

## Fields

| Field | Summary |
|:-----|:--------|
| `Average` | Takes the average (mean) value from the input region. |
| `L2Norm` | Calculates the L2 norm (Euclidean norm) of the values in the input region. |
| `Max` | Takes the maximum value from the input region. |


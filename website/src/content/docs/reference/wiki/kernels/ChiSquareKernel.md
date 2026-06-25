---
title: "ChiSquareKernel<T>"
description: "Implements the Chi-Square kernel function for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Chi-Square kernel function for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Chi-Square kernel is especially good at comparing data that represents counts or frequencies,
like how often words appear in documents or how many pixels of each color are in an image.

## How It Works

The Chi-Square kernel is based on the Chi-Square distance, which is particularly effective
for histogram data (such as image histograms, text frequency counts, etc.). It is commonly
used in computer vision and document classification tasks.

Think of the Chi-Square kernel as a specialized tool for comparing "how much" of something exists
in two different samples. It's particularly sensitive to differences in smaller values, which makes
it good at detecting subtle patterns in your data. For example, it might notice that two documents
use rare words in similar ways, even if their common words are different.

This kernel works best when your data contains only non-negative values, such as counts,
frequencies, or proportions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChiSquareKernel` | Initializes a new instance of the Chi-Square kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Chi-Square kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |


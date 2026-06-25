---
title: "SVMOptions<T>"
description: "Configuration options for Support Vector Machine classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Support Vector Machine classifiers.

## For Beginners

SVMs are like drawing the best possible line between classes!

Imagine you have red and blue dots on paper. You want to draw a line that:

1. Separates the red dots from the blue dots
2. Stays as far as possible from both sets of dots

The "margin" is the gap between the line and the nearest dots on each side.
SVMs find the line that maximizes this margin.

Key concepts:

- C parameter: How much to penalize misclassifications (higher = stricter)
- Kernel: How to measure similarity between points (linear, RBF, polynomial)
- Support vectors: The dots closest to the line that define its position

SVMs work great when:

- You have clear separation between classes
- You have many features but not tons of samples
- You need a robust classifier

## How It Works

Support Vector Machines (SVMs) find the optimal hyperplane that maximizes the margin
between classes. They are particularly effective in high-dimensional spaces and
can handle non-linear classification using kernel functions.

## Properties

| Property | Summary |
|:-----|:--------|
| `C` | Gets or sets the regularization parameter C. |
| `CacheSize` | Gets or sets the cache size in MB for kernel computations. |
| `Coef0` | Gets or sets the independent term (coef0) in kernel function. |
| `Degree` | Gets or sets the degree for polynomial kernel. |
| `Gamma` | Gets or sets the kernel coefficient gamma. |
| `Kernel` | Gets or sets the kernel type. |
| `MaxIterations` | Gets or sets the maximum number of iterations. |
| `OneVsRest` | Gets or sets whether to use one-vs-rest or one-vs-one for multi-class. |
| `Probability` | Gets or sets whether to calculate probability estimates. |
| `Shrinking` | Gets or sets whether to use shrinking heuristic. |
| `Tolerance` | Gets or sets the tolerance for stopping criterion. |


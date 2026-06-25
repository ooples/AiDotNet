---
title: "KernelType"
description: "Specifies different kernel functions used in machine learning algorithms like Support Vector Machines (SVMs)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies different kernel functions used in machine learning algorithms like Support Vector Machines (SVMs).

## For Beginners

A kernel is a special mathematical function that helps machine learning algorithms 
work with complex data. Think of kernels as "similarity measures" between data points.

Imagine you have data that can't be easily separated by a straight line. Kernels help by 
transforming your data into a form where patterns become more obvious - like lifting a 
2D drawing into 3D space where you can see separations more clearly.

Kernels are commonly used in:

- Support Vector Machines (SVMs)
- Kernel regression
- Principal Component Analysis (PCA)
- Clustering algorithms

Different kernels work better for different types of data and problems. Choosing the right 
kernel can significantly improve your model's performance.

## Fields

| Field | Summary |
|:-----|:--------|
| `Laplacian` | A kernel function that uses the negative exponential of the L1 distance between points. |
| `Linear` | The simplest kernel function that represents the standard dot product in the input space. |
| `Polynomial` | A flexible kernel that raises the dot product of features to a specified power. |
| `Precomputed` | Indicates that the kernel matrix is precomputed rather than being computed on-the-fly. |
| `RBF` | Radial Basis Function kernel that measures similarity based on distance in a high-dimensional space. |
| `Sigmoid` | A kernel function based on the hyperbolic tangent, similar to neural network activation functions. |


---
title: "GridKernel<T>"
description: "Grid Kernel for exploiting Kronecker structure in regularly-spaced data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Grid Kernel for exploiting Kronecker structure in regularly-spaced data.

## For Beginners

When your input data lies on a regular grid (like pixels in an image,
or time series sampled at regular intervals), the kernel matrix has special structure
that can be exploited for massive computational savings.

For data on a D-dimensional grid with n_1 × n_2 × ... × n_D points:
K = K_1 ⊗ K_2 ⊗ ... ⊗ K_D (Kronecker product)

This means:

- Storage: O(sum of n_i) instead of O((product of n_i)²)
- Matrix-vector multiply: O(product of n_i × sum of n_i) instead of O((product of n_i)²)

Example: 100×100 image grid

- Full kernel: 10,000 × 10,000 = 100M entries
- Grid kernel: 100 + 100 = 200 entries (500,000× less memory!)

Limitations:

- Only works for data on regular grids
- All dimensions must use the same base kernel (but can have different lengthscales)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GridKernel(IKernelFunction<>[],Double[][])` | Initializes a Grid Kernel with specified kernels per dimension. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GridSizes` | Gets the grid sizes along each dimension. |
| `NumDimensions` | Gets the number of dimensions. |
| `TotalGridPoints` | Gets the total number of grid points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the kernel value between two grid points. |
| `EigenDecomposition(Matrix<>)` | Simple eigendecomposition for symmetric matrices. |
| `GetEigenvalues` | Gets the eigenvalues of the full Kronecker kernel matrix. |
| `KroneckerMultiply(Vector<>)` | Performs efficient matrix-vector product K * v using Kronecker structure. |
| `LogDeterminant` | Computes log-determinant using Kronecker structure. |
| `Precompute` | Precomputes the 1D kernel matrices and their eigendecompositions. |
| `WithRBF(Double[][],Double[])` | Creates a Grid Kernel with RBF kernels for all dimensions. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dimensionKernelMatrices` | Precomputed 1D kernel matrices for each dimension. |
| `_dimensionKernels` | Base kernels for each dimension. |
| `_eigenvalues` | Eigenvalues of each 1D kernel matrix. |
| `_eigenvectors` | Eigenvectors of each 1D kernel matrix. |
| `_gridCoordinates` | Grid coordinates along each dimension. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_precomputed` | Whether precomputation has been done. |


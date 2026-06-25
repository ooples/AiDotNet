---
title: "GridInterpolationKernel<T>"
description: "Grid Interpolation Kernel (KISS-GP) for scalable Gaussian Process inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Grid Interpolation Kernel (KISS-GP) for scalable Gaussian Process inference.

## For Beginners

KISS-GP (Kernel Interpolation for Scalable Structured GPs) combines
inducing points with grid structure for highly scalable GP inference.

The key insight: Place inducing points on a regular grid, then use:

1. Interpolation to map data points to the grid
2. Kronecker/Toeplitz structure of the grid for fast computations

K ≈ W × K_grid × W'

Where:

- W is a sparse interpolation matrix (each data point → nearby grid points)
- K_grid has Kronecker structure (efficient to work with)

Complexity for N data points, M grid points:

- Standard GP: O(N³)
- Inducing points: O(NM² + M³)
- KISS-GP: O(N + M log M) using FFT!

This enables GPs with millions of data points on commodity hardware.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GridInterpolationKernel(IKernelFunction<>,Double[][],Int32)` | Initializes a Grid Interpolation Kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseKernel` | Gets the base kernel. |
| `NumDimensions` | Gets the number of dimensions. |
| `TotalGridPoints` | Gets the total number of grid points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the kernel value between two points. |
| `ComputeInterpolationMatrix(Matrix<>)` | Computes the interpolation matrix W for a set of data points. |
| `FastGridMultiply(Vector<>)` | Performs fast matrix-vector product using Kronecker-Toeplitz structure. |
| `Get1DInterpolationWeights(Double,Double[],Int32)` | Gets 1D cubic interpolation weights. |
| `GetInterpolationWeights(Vector<>)` | Gets interpolation weights for a point. |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix. |
| `IndexToGridPoint(Int32)` | Converts flat index to grid point. |
| `KissGpMultiply(Vector<>,Int32[][],Double[][])` | Performs full KISS-GP kernel-vector product: W × K_grid × W' × v. |
| `Precompute` | Precomputes Toeplitz structure for efficient matrix-vector products. |
| `ToeplitzMultiply(Vector<>,Double[])` | Toeplitz matrix-vector multiply (direct, O(n²) - could use FFT for O(n log n)). |
| `WithUniformGrid(IKernelFunction<>,ValueTuple<Double,Double>[],Int32,Int32)` | Creates a Grid Interpolation Kernel with automatic grid spacing. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseKernel` | The base kernel. |
| `_gridCoordinates` | Grid coordinates along each dimension. |
| `_gridSizes` | Number of grid points per dimension. |
| `_interpolationOrder` | Number of nearest grid points to interpolate from per dimension. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_precomputed` | Whether precomputation is done. |
| `_toeplitzColumns` | Precomputed Toeplitz column for each dimension (for stationary kernels). |


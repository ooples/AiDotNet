---
title: "RFFKernel<T>"
description: "Random Fourier Features (RFF) kernel for scalable approximation of shift-invariant kernels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Random Fourier Features (RFF) kernel for scalable approximation of shift-invariant kernels.

## For Beginners

Random Fourier Features is a clever technique for making Gaussian Processes
scale to large datasets. Instead of computing the full N×N kernel matrix (which has O(N²) memory
and O(N³) computation), RFF approximates the kernel with D random features.

The key insight (Bochner's theorem): Any shift-invariant kernel can be written as the
expectation of random cosine features:

k(x, x') ≈ (1/D) Σᵢ cos(ωᵢ·x + bᵢ) × cos(ωᵢ·x' + bᵢ)
= φ(x)ᵀ × φ(x')

Where:

- ωᵢ are random frequencies drawn from the kernel's spectral density
- bᵢ are random phases drawn uniformly from [0, 2π]
- φ(x) is a D-dimensional feature map

This transforms the GP kernel computation into a linear model:

- Instead of O(N³) for exact GP, we get O(ND² + D³)
- Memory goes from O(N²) to O(ND)

For D ≈ 1000-10000, the approximation is usually very good.

## How It Works

Applications:

- Large-scale GP regression (N > 10,000 points)
- Online/streaming GP updates
- Deep kernel learning (combining with neural networks)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RFFKernel(Int32,Int32,RFFKernel<>.RFFKernelType,Double,Double,Nullable<Int32>)` | Initializes a new RFF kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDim` | Gets the input dimensionality. |
| `KernelType` | Gets the kernel type being approximated. |
| `LengthScale` | Gets the length scale. |
| `NumFeatures` | Gets the number of random features. |
| `OutputScale` | Gets the output scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the approximate kernel value between two vectors. |
| `EstimateApproximationError(Matrix<>,IKernelFunction<>)` | Estimates the approximation quality by comparing to exact kernel on test points. |
| `GetFeatureMatrix(Matrix<>)` | Computes the feature matrix for multiple input points. |
| `GetFeatures(Vector<>)` | Computes the random Fourier feature map for an input vector. |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix as a vector. |
| `SampleCauchy(Random)` | Samples from a Cauchy distribution. |
| `SampleFrequencies(Random)` | Samples random frequencies from the spectral density of the kernel. |
| `SampleGaussian(Random)` | Samples from a standard Gaussian distribution using Box-Muller transform. |
| `SamplePhases(Random)` | Samples random phases uniformly from [0, 2π]. |
| `SampleStudentT(Random,Double)` | Samples from a Student-t distribution with given degrees of freedom. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_frequencies` | The random frequency vectors (ω). |
| `_inputDim` | Input dimensionality. |
| `_kernelType` | The kernel type being approximated. |
| `_lengthScale` | Length scale parameter. |
| `_numFeatures` | Number of random features. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_outputScale` | Output scale parameter. |
| `_phases` | The random phase shifts (b). |


---
title: "GPUExplainerHelper<T>"
description: "Provides GPU acceleration for interpretability explainers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability.Helpers`

Provides GPU acceleration for interpretability explainers.

## For Beginners

This helper accelerates interpretability computations using GPU hardware.
Many explanation methods (SHAP, Integrated Gradients, etc.) require computing many predictions
or gradients - operations that are perfect for GPU parallelization.

The helper provides:

1. **Batch Prediction**: Process many inputs simultaneously on GPU
2. **Parallel Coalition Processing**: For SHAP-style algorithms
3. **GPU Matrix Operations**: Fast linear algebra for solving attribution problems
4. **Automatic Fallback**: Falls back to CPU if no GPU is available

Benefits of GPU acceleration for explainers:

- 10-100x speedup for batch predictions
- Enables real-time explanations for complex models
- Makes ensemble/sample-based methods practical for large models

Usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GPUExplainerHelper(Nullable<Int32>)` | Initializes a new GPU explainer helper using CPU parallel processing. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DeviceInfo` | Gets information about the GPU device. |
| `IsGPUEnabled` | Gets whether GPU acceleration is available and enabled. |
| `MaxParallelism` | Gets the maximum parallelism level for CPU fallback operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchComputeGradients(Func<Vector<>,Int32,Vector<>>,Matrix<>,Int32)` | Computes gradients for multiple inputs in parallel using backpropagation. |
| `BatchPredict(Func<Matrix<>,Vector<>>,Matrix<>)` | Processes predictions for multiple inputs in parallel. |
| `BatchPredictCPU(Func<Matrix<>,Vector<>>,Matrix<>)` | CPU parallel batch prediction. |
| `BatchPredictGPU(Func<Matrix<>,Vector<>>,Matrix<>)` | GPU-accelerated batch prediction. |
| `ComputeCoalitionPredictions(Func<Matrix<>,Vector<>>,Vector<>,List<Boolean[]>,Matrix<>,Int32)` | Computes predictions for multiple coalitions in parallel (for SHAP-style algorithms). |
| `ComputeIntegratedGradientsParallel(Func<Vector<>,Int32,Vector<>>,Vector<>,Vector<>,Int32,Int32)` | Computes Integrated Gradients using parallel path integration. |
| `ComputePermutedPredictions(Func<Matrix<>,Vector<>>,Matrix<>,Int32,Int32,Nullable<Int32>)` | Processes feature permutations in parallel for permutation importance. |
| `CreateCPUOnly(Nullable<Int32>)` | Creates a CPU-only helper (no GPU acceleration). |
| `CreateWithAutoDetect` | Creates a GPU explainer helper with automatic GPU detection. |
| `Dispose` | Disposes GPU resources. |
| `ExtractRows(Matrix<>,Int32,Int32)` | Extracts a contiguous block of rows from a matrix. |
| `SolveLinearSystem(Double[0:,0:],Double[])` | Solves a linear system Ax = b using Gaussian elimination with partial pivoting. |
| `SolveWeightedLeastSquares(Matrix<>,Vector<>,Vector<>)` | Solves weighted least squares using GPU-accelerated matrix operations. |


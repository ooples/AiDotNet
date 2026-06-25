---
title: "MultiTaskGaussianProcess<T>"
description: "Implements a Multi-Task Gaussian Process for modeling multiple correlated outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a Multi-Task Gaussian Process for modeling multiple correlated outputs.

## For Beginners

A Multi-Task GP models multiple related output variables simultaneously,
learning the correlations between tasks to improve predictions for all of them.

Example scenarios:

- Predicting temperature, humidity, and pressure at weather stations (related measurements)
- Forecasting sales across multiple product lines (correlated markets)
- Modeling grades across subjects for students (abilities correlate)

Why use Multi-Task GP instead of separate GPs?

1. **Information sharing**: If task A has lots of data and task B has little,

the Multi-Task GP can use A's data to help predict B

2. **Correlation modeling**: Learns how tasks relate (e.g., when temperature rises,

ice cream sales increase)

3. **Better uncertainty**: More accurate confidence intervals by considering

task relationships

The model uses a coregionalization approach:

- A base kernel captures input similarity
- A task correlation matrix captures how tasks relate
- The combined kernel is their product (Kronecker structure)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiTaskGaussianProcess(IKernelFunction<>,Int32,Double,Boolean,MatrixDecompositionType)` | Initializes a new Multi-Task Gaussian Process. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildCombinedKernel` | Builds the combined covariance matrix using Kronecker structure. |
| `ComputeAlpha` | Computes the alpha vector for predictions. |
| `Fit(Matrix<>,Matrix<>)` | Trains the Multi-Task GP on the provided data. |
| `Fit(Matrix<>,Vector<>)` | IFullModel compliance: Fit with single-output vector (uses first task only). |
| `GetTaskCorrelations` | Gets the learned task correlation matrix. |
| `LearnTaskCorrelations` | Learns the task correlation matrix from the data. |
| `Predict(Vector<>)` | IFullModel compliance: Predict single point returning first task's mean. |
| `PredictMultiTask(Vector<>)` | Predicts all task outputs for a new input point. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel for this multi-task GP. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_K` | The combined kernel matrix. |
| `_X` | The training input data. |
| `_Y` | The training target values (multi-output). |
| `_alpha` | The alpha vector for predictions (K^(-1) * y). |
| `_decompositionType` | Matrix decomposition method. |
| `_kernel` | The base kernel for input similarity. |
| `_learnTaskCorrelations` | Whether to learn task correlations from data. |
| `_noiseVariance` | Observation noise variance. |
| `_numOps` | Operations for numeric calculations. |
| `_numTasks` | The number of tasks (output dimensions). |
| `_taskCovCholesky` | Cholesky factor of the task covariance. |
| `_taskCovariance` | The task correlation matrix (B matrix in ICM/LMC models). |


---
title: "SparseVariationalGaussianProcess<T>"
description: "Implements a Sparse Variational Gaussian Process (SVGP) for scalable GP regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a Sparse Variational Gaussian Process (SVGP) for scalable GP regression.

## For Beginners

Standard Gaussian Processes are powerful but slow - they require O(n³)
computation time, making them impractical for datasets larger than a few thousand points.

SVGP solves this problem using two key ideas:

1. **Inducing Points**: Instead of using all training points, we use a smaller set of

"representative" points (inducing points) that summarize the data. If we have n training
points but only m inducing points (where m << n), computation becomes O(nm²) instead of O(n³).

2. **Variational Inference**: We approximate the true posterior distribution with a simpler

distribution (variational distribution) that's easier to work with. We optimize this
approximation to be as close as possible to the true posterior.

The result is a GP that can handle millions of data points while still providing
uncertainty estimates and probabilistic predictions.

## How It Works

When to use SVGP:

- Large datasets (thousands to millions of points)
- When you need mini-batch training (can't fit all data in memory)
- When you want uncertainty estimates but standard GP is too slow
- When inducing points can reasonably summarize your data

Limitations:

- The approximation quality depends on the number and placement of inducing points
- More hyperparameters to tune compared to standard GP
- May require more iterations to converge

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparseVariationalGaussianProcess(IKernelFunction<>,Int32,Double,Double,Int32,MatrixDecompositionType)` | Initializes a new instance of the SparseVariationalGaussianProcess class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddJitter(Matrix<>)` | Adds jitter to the diagonal for numerical stability. |
| `BackSolve(Matrix<>,Vector<>)` | Back-solves U * x = b for upper-triangular U. |
| `CalculateKernelMatrix(Matrix<>,Matrix<>)` | Calculates the kernel matrix between two sets of data points. |
| `CalculateKernelVector(Matrix<>,Vector<>)` | Calculates kernel values between a matrix of points and a single vector. |
| `ComputeELBO` | Computes the Evidence Lower Bound (ELBO) for the current variational approximation. |
| `ComputePredictiveMean(Matrix<>)` | Computes the predictive mean at training points. |
| `CreateIdentityMatrix(Int32)` | Creates an identity matrix of the specified size. |
| `Fit(Matrix<>,Vector<>)` | Trains the SVGP model using variational inference and gradient-based optimization. |
| `ForceSymmetric(Matrix<>)` | Forces a matrix to be exactly symmetric: K = (K + K^T) / 2. |
| `ForwardSolve(Matrix<>,Vector<>)` | Forward-solves L * x = b for lower-triangular L. |
| `OptimizeVariationalParameters` | Optimizes the variational parameters using gradient ascent on the ELBO. |
| `Predict(Vector<>)` |  |
| `SelectInducingPoints(Matrix<>,Int32)` | Selects inducing points from the training data using random sampling. |
| `UpdateKernel(IKernelFunction<>)` |  |
| `UpdateVariationalCovariance(Matrix<>,)` | Updates the variational covariance based on the data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_Kuu` | Kernel matrix between inducing points (Kuu). |
| `_LKuu` | Cholesky factor of Kuu for efficient computation. |
| `_X` | The matrix of input features from the training data. |
| `_decompositionType` | The method used for matrix decomposition in linear system solving. |
| `_inducingPoints` | The inducing points that summarize the training data. |
| `_kernel` | The kernel function that determines similarity between data points. |
| `_learningRate` | Learning rate for variational parameter updates. |
| `_maxIterations` | Number of optimization iterations. |
| `_noiseVariance` | The observation noise variance (likelihood parameter). |
| `_numInducingPoints` | The number of inducing points to use. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_variationalCovCholesky` | The Cholesky factor of the variational covariance matrix. |
| `_variationalMean` | The variational mean of the approximate posterior at inducing points. |
| `_y` | The vector of target values from the training data. |


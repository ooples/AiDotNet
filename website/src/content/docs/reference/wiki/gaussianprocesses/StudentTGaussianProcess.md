---
title: "StudentTGaussianProcess<T>"
description: "Implements a Gaussian Process with Student-t likelihood for robust regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a Gaussian Process with Student-t likelihood for robust regression.

## For Beginners

Standard Gaussian Processes assume Gaussian (normal) noise,
which is sensitive to outliers. A single outlier can drastically affect predictions.

The Student-t GP uses a heavy-tailed Student-t distribution instead:

- More tolerant of outliers (they have less influence)
- The "degrees of freedom" parameter (ν) controls robustness:
- ν = 1: Cauchy distribution (very robust, heavy tails)
- ν = 4-5: Good balance of robustness and efficiency
- ν → ∞: Approaches Gaussian (standard GP)

This uses Expectation Propagation (EP) for approximate inference since
the Student-t likelihood makes exact inference intractable.

## How It Works

When to use:

- Data with potential outliers
- Sensor data with occasional erroneous readings
- Financial data with market anomalies
- Any regression where robustness to bad data points is important

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StudentTGaussianProcess(IKernelFunction<>,Double,Double,Int32,Double,Double,Double)` | Initializes a new Student-t Gaussian Process. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DegreesOfFreedom` | Gets the degrees of freedom. |
| `Kernel` | Gets the kernel function. |
| `Scale` | Gets the noise scale parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CholeskyDecomposition(Matrix<>)` | Performs Cholesky decomposition. |
| `ComputeInverseFromCholesky(Matrix<>)` | Computes matrix inverse from Cholesky factor. |
| `ComputeKernelMatrix(Matrix<>,Matrix<>)` | Computes the kernel matrix between two sets of points. |
| `ComputeLogMarginalLikelihood` | Computes the log marginal likelihood approximation. |
| `ComputeTiltedMoments(Double,Double,Double)` | Computes moments of the tilted distribution via numerical integration. |
| `CopyMatrix(Matrix<>)` | Creates a copy of a matrix. |
| `Fit(Matrix<>,Vector<>)` | Trains the Student-t GP using Expectation Propagation. |
| `GaussHermiteQuadrature(Int32,Double[],Double[])` | Gauss-Hermite quadrature nodes and weights. |
| `GetOutlierIndices(Double)` | Identifies outliers in the training data. |
| `GetOutlierWeights` | Gets the outlier weights after training. |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix as a vector. |
| `LogGamma(Double)` | Computes log gamma function. |
| `Predict(Vector<>)` | Predicts the mean and variance for a single input point. |
| `PredictBatch(Matrix<>)` | Makes predictions with uncertainty estimates for multiple points. |
| `SolveLowerTriangular(Matrix<>,Vector<>)` | Solves Lx = b for x where L is lower triangular. |
| `SolveUpperTriangular(Matrix<>,Vector<>)` | Solves Ux = b for x where U is upper triangular. |
| `StudentTLogPdf(Double,Double,Double)` | Computes the log PDF of the Student-t distribution. |
| `Transpose(Matrix<>)` | Transposes a matrix. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel function. |
| `UpdatePosterior` | Updates the posterior distribution from site parameters. |
| `UpdateSite(Int32)` | Updates a single EP site. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_K` | Prior covariance matrix. |
| `_X` | Training input data. |
| `_damping` | Damping factor for EP updates. |
| `_isTrained` | Whether the model has been trained. |
| `_jitter` | Small regularization constant. |
| `_kernel` | The kernel function. |
| `_maxIterations` | Maximum number of EP iterations. |
| `_nu` | Degrees of freedom for Student-t distribution. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_posteriorCov` | Approximate posterior covariance. |
| `_posteriorMean` | Approximate posterior mean. |
| `_scale` | Scale parameter for Student-t noise. |
| `_siteNaturalMeans` | Site natural parameters (precision * mean). |
| `_sitePrecisions` | Site natural parameters (precision). |
| `_tolerance` | Convergence tolerance. |
| `_weights` | Outlier weights (downweights outliers). |
| `_y` | Training target values. |


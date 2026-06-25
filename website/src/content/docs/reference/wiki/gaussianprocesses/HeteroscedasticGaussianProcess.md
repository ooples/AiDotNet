---
title: "HeteroscedasticGaussianProcess<T>"
description: "Implements a Gaussian Process with input-dependent (heteroscedastic) noise levels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a Gaussian Process with input-dependent (heteroscedastic) noise levels.

## For Beginners

Standard Gaussian Processes assume the same level of noise
everywhere in your data. But real-world data often has varying uncertainty:

- Sensor readings may be more accurate in certain conditions
- Financial data has varying volatility
- Experimental measurements may be more precise in some ranges

The Heteroscedastic GP models this by learning a separate GP for the noise variance,
allowing it to capture input-dependent uncertainty.

This uses the "Most Likely Heteroscedastic GP" (MLHGP) approach:

1. A primary GP models the mean function f(x)
2. A secondary GP models the log noise variance g(x) = log(σ²(x))
3. The two GPs are jointly optimized

The noise at any point x is: σ²(x) = exp(g(x))
This exponential ensures the variance is always positive.

## How It Works

When to use:

- Data with varying noise levels
- Sensor fusion with different accuracy sensors
- Financial modeling with time-varying volatility
- Any regression where uncertainty varies with input

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeteroscedasticGaussianProcess(IKernelFunction<>,IKernelFunction<>,Double,Double,Int32,Double,Double)` | Initializes a new heteroscedastic Gaussian Process. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MeanKernel` | Gets the kernel function used for the mean GP. |
| `NoiseKernel` | Gets the kernel function used for the noise GP. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAdaptiveJitter(Matrix<>)` | Adds adaptive jitter to make the matrix well-conditioned for Cholesky decomposition. |
| `CholeskyDecomposition(Matrix<>)` | Performs Cholesky decomposition of a symmetric positive-definite matrix. |
| `ComputeNegativeLogLikelihood` | Computes the negative log marginal likelihood. |
| `Fit(Matrix<>,Vector<>)` | Trains the heteroscedastic GP on the given data. |
| `FitMeanGP` | Fits the mean GP with current noise estimates. |
| `GetNoiseVariances` | Gets the learned noise variances at training points. |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix as a vector. |
| `GetTotalUncertainty(Matrix<>)` | Gets the total predictive uncertainty (epistemic + aleatoric). |
| `Predict(Vector<>)` | Predicts the mean and variance for a single input point. |
| `PredictWithUncertainty(Matrix<>)` | Makes predictions with uncertainty estimates. |
| `SolveLowerTriangular(Matrix<>,Vector<>)` | Solves Lx = b for x where L is lower triangular. |
| `SolveUpperTriangular(Matrix<>,Vector<>)` | Solves Ux = b for x where U is upper triangular. |
| `Transpose(Matrix<>)` | Transposes a matrix. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel function used for the mean GP. |
| `UpdateNoiseLatentValues` | Updates noise latent values using the current mean GP. |
| `UpdateNoiseVariances` | Updates the noise variance values from latent values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_L` | Cholesky decomposition of mean GP covariance matrix. |
| `_X` | Training input data. |
| `_alpha` | Alpha coefficients for mean predictions. |
| `_isTrained` | Whether the model has been trained. |
| `_jitter` | Small regularization constant. |
| `_maxIterations` | Number of EM iterations for joint optimization. |
| `_meanKernel` | The kernel function for the mean GP. |
| `_noiseKernel` | The kernel function for the noise GP. |
| `_noiseLatentValues` | Noise GP latent values (log noise variance). |
| `_noisePriorMean` | Prior mean for noise GP (log scale). |
| `_noisePriorVariance` | Prior variance for noise GP. |
| `_noiseVariances` | Learned noise variances at training points. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_tolerance` | Convergence tolerance for EM algorithm. |
| `_y` | Training target values. |


---
title: "GPWithMCMC<T>"
description: "Gaussian Process with Markov Chain Monte Carlo inference for full Bayesian treatment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Gaussian Process with Markov Chain Monte Carlo inference for full Bayesian treatment.

## For Beginners

Standard GP makes point estimates of hyperparameters (kernel lengthscale,
output variance, noise variance). MCMC provides a fully Bayesian treatment by sampling from
the posterior distribution of hyperparameters, giving better uncertainty quantification.

Instead of finding a single "best" lengthscale, MCMC explores many plausible lengthscales
and averages predictions across all of them. This is more robust when:

- You have limited data
- The hyperparameters are uncertain
- You need accurate uncertainty estimates

The implementation uses Slice Sampling, which automatically adapts to the target distribution
without requiring careful tuning of step sizes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GPWithMCMC(IKernelFunction<>,Int32,Int32,Int32,Double,Double,Double,Double,Nullable<Int32>)` | Initializes a GP with MCMC inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsTrained` | Gets whether the GP is trained. |
| `Kernel` | Gets the kernel function. |
| `NumStoredSamples` | Gets the number of stored MCMC samples. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildKernelMatrix(Double,Double,Double)` | Builds kernel matrix with given hyperparameters. |
| `CholeskyDecomposition(Matrix<>)` | Cholesky decomposition of a matrix. |
| `ComputeCrossCovariance(Vector<>,Double,Double)` | Computes cross-covariance k(x, X). |
| `ComputeLogPosterior(Double[])` | Computes log posterior = log likelihood + log prior. |
| `Fit(Matrix<>,Vector<>)` | Fits the GP to training data using MCMC sampling. |
| `GetPosteriorStatistics` | Computes posterior statistics for hyperparameters. |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix. |
| `GetSamples` | Gets the MCMC samples [lengthscale, outputVariance, noiseVariance]. |
| `LogNormalPdf(Double,Double,Double)` | Log of normal PDF. |
| `PrecomputeForSamples` | Precomputes Cholesky factors and alpha vectors for all samples. |
| `Predict(Vector<>)` | Predicts at a single test point with full posterior averaging. |
| `PredictBatch(Matrix<>)` | Predicts at multiple test points. |
| `SliceSample(Double[],Int32,Double)` | Slice sampling for one parameter. |
| `SolveLowerTriangular(Matrix<>,Vector<>)` | Solves Lx = b for lower triangular L. |
| `SolveUpperTriangular(Matrix<>,Vector<>)` | Solves L'x = b for lower triangular L (i.e., solves upper triangular system). |
| `StdDev(Double[])` | Computes standard deviation. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel (not supported for MCMC GP). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_X` | Training input data. |
| `_alphaVectors` | Precomputed alpha vectors for each sample. |
| `_burnIn` | Number of burn-in samples to discard. |
| `_choleskyFactors` | Precomputed Cholesky factors for each sample (for efficiency). |
| `_isTrained` | Whether the model has been trained. |
| `_kernel` | The base kernel function. |
| `_logLengthscalePriorMean` | Prior mean for log-lengthscale. |
| `_logLengthscalePriorStd` | Prior std for log-lengthscale. |
| `_logVariancePriorMean` | Prior mean for log-variance. |
| `_logVariancePriorStd` | Prior std for log-variance. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_numSamples` | Number of MCMC samples to use. |
| `_random` | Random generator for MCMC. |
| `_samples` | MCMC samples of hyperparameters [lengthscale, outputVariance, noiseVariance]. |
| `_thinning` | Thinning factor (keep every nth sample). |
| `_y` | Training output data, standardized as `(y - mean(y)) / std(y)`. |
| `_yMean` | Mean of the original training targets, used to de-center predictions. |
| `_yStd` | Standard deviation of the original training targets for output variance scaling. |


---
title: "VariationalGaussianProcess<T>"
description: "Implements a Variational Gaussian Process (VGP) using variational inference for exact data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a Variational Gaussian Process (VGP) using variational inference for exact data.

## For Beginners

The Variational Gaussian Process (VGP) is a probabilistic model that uses
variational inference to approximate the posterior distribution over functions.

Unlike SVGP which uses inducing points for scalability, VGP works with all training points
but uses variational inference to handle non-Gaussian likelihoods (like for classification
or robust regression with non-Gaussian noise).

Key differences from standard GP:

- Standard GP: Assumes Gaussian likelihood, has closed-form solution
- VGP: Can handle any likelihood, uses optimization to find approximate posterior

When to use VGP:

- When you have non-Gaussian likelihoods (classification, count data, etc.)
- When you want uncertainty quantification with flexible likelihood models
- When your dataset is small enough to use all points (up to ~5000 points)

For large datasets with Gaussian likelihood, use SparseVariationalGaussianProcess instead.

## How It Works

**Implementation Note:** The ELBO (Evidence Lower Bound) computation uses a simplified
trace approximation (S ≈ K) for numerical stability. This approximation may affect the
absolute ELBO values, making them unreliable for cross-model comparison. For hyperparameter
optimization, consider using the prediction accuracy on a held-out validation set instead.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationalGaussianProcess(IKernelFunction<>,VGPLikelihood,Double,Double,Int32,Double,MatrixDecompositionType)` | Initializes a new instance of the VariationalGaussianProcess class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddJitter(Matrix<>)` | Adds jitter to diagonal for numerical stability. |
| `CalculateKernelMatrix(Matrix<>,Matrix<>)` | Calculates the kernel matrix between two sets of points. |
| `CalculateKernelVector(Matrix<>,Vector<>)` | Calculates kernel values between a matrix and a vector. |
| `ComputeELBO` | Computes the Evidence Lower Bound (ELBO) for the current variational approximation. |
| `ComputeLikelihoodGradient` | Computes the gradient of the expected log-likelihood. |
| `CreateIdentityMatrix(Int32)` | Creates an identity matrix. |
| `CreateScaledIdentityMatrix(Int32,Double)` | Creates a scaled identity matrix. |
| `Fit(Matrix<>,Vector<>)` | Trains the VGP model using variational inference. |
| `LogFactorial(Int32)` | Computes log factorial for Poisson likelihood. |
| `OptimizeGaussianLikelihood` | Optimizes variational parameters for Gaussian likelihood (closed-form). |
| `OptimizeNonGaussianLikelihood` | Optimizes variational parameters for non-Gaussian likelihoods using gradient ascent. |
| `Predict(Vector<>)` |  |
| `UpdateKernel(IKernelFunction<>)` |  |
| `UpdateVariationalCovariance` | Updates the variational covariance based on likelihood curvature. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_K` | The kernel matrix computed from training data. |
| `_KPlusNoise` | The well-conditioned system (K + σ²I) from the closed-form Gaussian likelihood fit. |
| `_LK` | Cholesky factor of the kernel matrix for efficient computation. |
| `_X` | The matrix of input features from the training data. |
| `_alpha` | Posterior weights α = (K + σ²I)⁻¹·y from the closed-form Gaussian likelihood fit, used for the stable predictive mean k*ᵀα. |
| `_decompositionType` | The method used for matrix decomposition. |
| `_kernel` | The kernel function that determines similarity between data points. |
| `_learningRate` | Learning rate for optimization. |
| `_likelihood` | The likelihood type for the model. |
| `_maxIterations` | Maximum number of optimization iterations. |
| `_noiseVariance` | The observation noise variance. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_tolerance` | Convergence tolerance for optimization. |
| `_variationalCovCholesky` | The Cholesky factor of the variational covariance matrix. |
| `_variationalMean` | The variational mean of the approximate posterior. |
| `_y` | The vector of target values from the training data. |


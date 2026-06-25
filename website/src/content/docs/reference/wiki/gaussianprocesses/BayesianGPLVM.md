---
title: "BayesianGPLVM<T>"
description: "Implements the Bayesian Gaussian Process Latent Variable Model (Bayesian GPLVM)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements the Bayesian Gaussian Process Latent Variable Model (Bayesian GPLVM).

## For Beginners

The Bayesian GPLVM is a powerful technique for dimensionality reduction
and learning latent representations of data. Unlike PCA which finds linear projections,
GPLVM finds nonlinear latent spaces using Gaussian Processes.

Key concepts:

1. **Latent Space**: A lower-dimensional representation where data lives
2. **Mapping**: A GP maps from latent space to observed data
3. **Uncertainty**: We maintain uncertainty over both the mapping and the latent points

Applications:

- Visualizing high-dimensional data (like t-SNE but probabilistic)
- Finding meaningful low-dimensional representations
- Handling missing data gracefully
- Interpolating between data points

## How It Works

**Mathematical Background:**
The model assumes observed data Y is generated from latent points X through a GP:
y_n = f(x_n) + ε, where f ~ GP(0, k)

The Bayesian approach places priors on the latent points X:
p(X) = Π_n N(x_n | 0, I)

We use variational inference to approximate the posterior p(X|Y).
The inducing points framework makes this scalable to large datasets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianGPLVM(IKernelFunction<>,Int32,Int32,Double,Double,Int32)` | Initializes a new instance of the Bayesian GPLVM. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsFitted` | Gets whether the model has been fitted. |
| `LatentDimensions` | Gets the latent dimension count. |
| `NumInducingPoints` | Gets the number of inducing points. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeELBO` | Computes the ELBO (Evidence Lower BOund) for model comparison. |
| `ComputeKernelMatrix(Matrix<>,Matrix<>)` | Computes the kernel matrix between two sets of points. |
| `ComputeLatentDistance(Int32,Int32)` | Computes the squared Euclidean distance between two latent points. |
| `FindNearestTrainingPoint(Vector<>)` | Finds the nearest training point to an observation. |
| `Fit(Matrix<>)` | Fits the Bayesian GPLVM to observed data. |
| `GetInducingPoint(Int32)` | Gets an inducing point as a vector. |
| `GetLatentMean` | Gets the learned latent representation (means). |
| `GetLatentPoint(Int32)` | Gets a latent point as a vector. |
| `GetLatentVariance` | Gets the learned latent representation (variances). |
| `GetObservedPoint(Int32)` | Gets an observed data point as a vector. |
| `InitializeInducingPoints` | Initializes inducing points by selecting from latent means. |
| `InitializeLatentPoints(Matrix<>)` | Initializes latent points using PCA. |
| `InvertMatrixSafe(Matrix<>)` | Safely inverts a matrix with regularization. |
| `OptimizeELBO` | Optimizes the ELBO using gradient descent. |
| `Reconstruct(Matrix<>)` | Reconstructs data from latent representations. |
| `ReconstructPoint(Vector<>)` | Reconstructs a data point from a latent representation. |
| `Transform(Matrix<>)` | Transforms new data points into the latent space. |
| `UpdateInducingPoints` | Updates inducing points using gradient descent. |
| `UpdateLatentPoints` | Updates latent points using gradient descent. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inducingPoints` | The inducing points in latent space (M x Q). |
| `_isFitted` | Whether the model has been fitted. |
| `_kernel` | The kernel function for the mapping from latent to observed space. |
| `_latentDimensions` | The number of latent dimensions. |
| `_latentMean` | The variational mean of the latent points (N x Q). |
| `_latentVariance` | The variational variance of the latent points (N x Q). |
| `_learningRate` | The learning rate for optimization. |
| `_maxIterations` | Maximum number of optimization iterations. |
| `_noiseVariance` | The observation noise variance. |
| `_numInducingPoints` | The number of inducing points for scalable inference. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_observedData` | The observed data (N x D). |


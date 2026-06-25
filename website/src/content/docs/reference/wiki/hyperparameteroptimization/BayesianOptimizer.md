---
title: "BayesianOptimizer<T, TInput, TOutput>"
description: "Implements Bayesian optimization for hyperparameter tuning using Gaussian Process regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Implements Bayesian optimization for hyperparameter tuning using Gaussian Process regression.

## How It Works

**For Beginners:** Bayesian optimization is a smart search strategy that learns from
previous trials to decide what to try next. Unlike grid or random search, it:

- Builds a model of how hyperparameters affect performance
- Uses this model to focus on promising regions
- Balances exploration (trying new areas) with exploitation (refining good areas)

This makes it much more efficient than random search, often finding good hyperparameters
in 10-20 trials instead of 100s.

Key components:

- Surrogate Model: A Gaussian Process that predicts performance for any hyperparameter combination
- Acquisition Function: Decides where to sample next based on predicted mean and uncertainty
- Sequential Optimization: Each trial informs the next, unlike parallel random search

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianOptimizer(Boolean,AcquisitionFunctionType,Int32,Double,Nullable<Int32>)` | Initializes a new instance of the BayesianOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ArrayToParameters(Double[],List<String>,HyperparameterSearchSpace)` | Converts a normalized array back to parameters. |
| `ComputeAcquisition(Double[])` | Computes the acquisition function value at a point. |
| `ComputeExpectedImprovement(Double,Double)` | Computes Expected Improvement (EI) acquisition function. |
| `ComputeLogMarginalLikelihood` | Computes the log marginal likelihood of the GP. |
| `ComputeLowerConfidenceBound(Double,Double)` | Computes Lower Confidence Bound (LCB) for minimization problems. |
| `ComputeProbabilityOfImprovement(Double,Double)` | Computes Probability of Improvement (PI) acquisition function. |
| `ComputeUpperConfidenceBound(Double,Double)` | Computes Upper Confidence Bound (UCB) acquisition function. |
| `DenormalizeParameter(Double,ParameterDistribution)` | Denormalizes a [0, 1] value back to parameter space. |
| `Erf(Double)` | Error function approximation. |
| `InvertMatrixCholesky(Double[0:,0:])` | Inverts a positive definite matrix using Cholesky decomposition. |
| `LogDeterminant(Double[0:,0:])` | Computes the log determinant of a matrix using Cholesky decomposition. |
| `NormalCdf(Double)` | Standard normal cumulative distribution function. |
| `NormalPdf(Double)` | Standard normal probability density function. |
| `NormalizeParameter(Object,ParameterDistribution)` | Normalizes a parameter value to [0, 1]. |
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameter configuration using Bayesian optimization. |
| `OptimizeAcquisitionFunction(List<String>)` | Optimizes the acquisition function to find the next sampling point. |
| `OptimizeGPHyperparameters` | Optimizes GP hyperparameters using marginal likelihood. |
| `ParametersToArray(Dictionary<String,Object>,List<String>,HyperparameterSearchSpace)` | Converts parameters to a normalized array [0, 1]. |
| `PredictGP(Double[])` | Predicts the mean and variance at a new point using the Gaussian Process. |
| `RBFKernel(Double[],Double[])` | Computes the RBF (Radial Basis Function) kernel between two points. |
| `SampleRandomPoint(HyperparameterSearchSpace)` | Samples a random point from the search space. |
| `SuggestNext(HyperparameterTrial<>)` | Suggests the next hyperparameter configuration using the acquisition function. |
| `UpdateCovarianceMatrix` | Updates the covariance matrix with current observations. |


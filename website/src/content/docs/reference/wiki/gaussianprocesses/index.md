---
title: "Gaussian Processes"
description: "All 29 public types in the AiDotNet.gaussianprocesses namespace, organized by kind."
section: "API Reference"
---

**29** public types in this namespace, organized by kind.

## Models & Types (24)

| Type | Summary |
|:-----|:--------|
| [`BayesianGPLVM<T>`](/docs/reference/wiki/gaussianprocesses/bayesiangplvm/) | Implements the Bayesian Gaussian Process Latent Variable Model (Bayesian GPLVM). |
| [`BernoulliLikelihood<T>`](/docs/reference/wiki/gaussianprocesses/bernoullilikelihood/) | Implements the Bernoulli likelihood for binary classification. |
| [`BetaLikelihood<T>`](/docs/reference/wiki/gaussianprocesses/betalikelihood/) | Beta Likelihood for Gaussian Processes with bounded outputs in [0, 1]. |
| [`ConstantMean<T>`](/docs/reference/wiki/gaussianprocesses/constantmean/) | Implements a constant mean function. |
| [`DeepGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/deepgaussianprocess/) | Implements a Deep Gaussian Process (DGP) with multiple stacked GP layers. |
| [`GPWithMCMC<T>`](/docs/reference/wiki/gaussianprocesses/gpwithmcmc/) | Gaussian Process with Markov Chain Monte Carlo inference for full Bayesian treatment. |
| [`GaussianLikelihood<T>`](/docs/reference/wiki/gaussianprocesses/gaussianlikelihood/) | Implements the Gaussian (Normal) likelihood for regression. |
| [`GaussianProcessClassifier<T>`](/docs/reference/wiki/gaussianprocesses/gaussianprocessclassifier/) | Implements a Gaussian Process Classifier using Laplace approximation for probabilistic classification. |
| [`HeteroscedasticGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/heteroscedasticgaussianprocess/) | Implements a Gaussian Process with input-dependent (heteroscedastic) noise levels. |
| [`HyperparameterOptimizer<T>`](/docs/reference/wiki/gaussianprocesses/hyperparameteroptimizer/) | Provides hyperparameter optimization for Gaussian Processes. |
| [`HyperparameterResult<T>`](/docs/reference/wiki/gaussianprocesses/hyperparameterresult/) | Represents a hyperparameter configuration and its score. |
| [`LinearMean<T>`](/docs/reference/wiki/gaussianprocesses/linearmean/) | Implements a linear mean function. |
| [`MultiOutputGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/multioutputgaussianprocess/) | A Gaussian Process model that can predict multiple output values simultaneously. |
| [`MultiTaskGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/multitaskgaussianprocess/) | Implements a Multi-Task Gaussian Process for modeling multiple correlated outputs. |
| [`PoissonLikelihood<T>`](/docs/reference/wiki/gaussianprocesses/poissonlikelihood/) | Implements the Poisson likelihood for count data. |
| [`PolynomialMean<T>`](/docs/reference/wiki/gaussianprocesses/polynomialmean/) | Implements a polynomial mean function. |
| [`SparseGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/sparsegaussianprocess/) | A sparse implementation of Gaussian Process regression that uses inducing points to reduce computational complexity. |
| [`SparseVariationalGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/sparsevariationalgaussianprocess/) | Implements a Sparse Variational Gaussian Process (SVGP) for scalable GP regression. |
| [`StandardGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/standardgaussianprocess/) | Implements a standard Gaussian Process regression model for making probabilistic predictions. |
| [`StudentTGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/studenttgaussianprocess/) | Implements a Gaussian Process with Student-t likelihood for robust regression. |
| [`StudentTLikelihood<T>`](/docs/reference/wiki/gaussianprocesses/studenttlikelihood/) | Implements the Student-t likelihood for robust regression. |
| [`VariationalGaussianProcess<T>`](/docs/reference/wiki/gaussianprocesses/variationalgaussianprocess/) | Implements a Variational Gaussian Process (VGP) using variational inference for exact data. |
| [`VariationalStrategies<T>`](/docs/reference/wiki/gaussianprocesses/variationalstrategies/) | Provides variational inference strategies for scalable Gaussian Process inference. |
| [`ZeroMean<T>`](/docs/reference/wiki/gaussianprocesses/zeromean/) | Implements a zero mean function. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`GaussianProcessBase<T>`](/docs/reference/wiki/gaussianprocesses/gaussianprocessbase/) | Abstract base class for Gaussian Process models that provides IFullModel compliance. |

## Interfaces (2)

| Type | Summary |
|:-----|:--------|
| [`ILikelihood<T>`](/docs/reference/wiki/gaussianprocesses/ilikelihood/) | Interface for likelihood functions in Gaussian Processes. |
| [`IMeanFunction<T>`](/docs/reference/wiki/gaussianprocesses/imeanfunction/) | Interface for mean functions in Gaussian Processes. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`OptimizationMethod<T>`](/docs/reference/wiki/gaussianprocesses/optimizationmethod/) | The optimization method to use. |
| [`VGPLikelihood`](/docs/reference/wiki/gaussianprocesses/vgplikelihood/) | Specifies the likelihood function type for Variational Gaussian Process. |


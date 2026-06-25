---
title: "Optimizers"
description: "All 43 public types in the AiDotNet.optimizers namespace, organized by kind."
section: "API Reference"
---

**43** public types in this namespace, organized by kind.

## Models & Types (40)

| Type | Summary |
|:-----|:--------|
| [`ADMMOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/admmoptimizer/) | Implements the Alternating Direction Method of Multipliers (ADMM) optimization algorithm. |
| [`AMSGradOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/amsgradoptimizer/) | Implements the AMSGrad optimization algorithm, an improved version of Adam optimizer. |
| [`AdaDeltaOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/adadeltaoptimizer/) | Implements the AdaDelta optimization algorithm for training neural networks and other machine learning models. |
| [`AdaMaxOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/adamaxoptimizer/) | Represents an AdaMax optimizer, an extension of Adam that uses the infinity norm. |
| [`AdagradOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/adagradoptimizer/) | Represents an Adagrad (Adaptive Gradient) optimizer for gradient-based optimization. |
| [`Adam8BitOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/adam8bitoptimizer/) | Implements an 8-bit quantized Adam optimizer that reduces memory usage by storing optimizer states in 8-bit format. |
| [`AdamOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/adamoptimizer/) | Implements the Adam (Adaptive Moment Estimation) optimization algorithm for gradient-based optimization. |
| [`AdamWOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/adamwoptimizer/) | Implements the AdamW (Adam with decoupled Weight decay) optimization algorithm. |
| [`AntColonyOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/antcolonyoptimizer/) | Implements the Ant Colony Optimization algorithm for solving optimization problems. |
| [`BFGSOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/bfgsoptimizer/) | Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm. |
| [`BayesianOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/bayesianoptimizer/) | Represents a Bayesian Optimizer for optimization problems. |
| [`CMAESOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/cmaesoptimizer/) | Implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm. |
| [`ConjugateGradientOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/conjugategradientoptimizer/) | Implements the Conjugate Gradient optimization algorithm for numerical optimization problems. |
| [`CoordinateDescentOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/coordinatedescentoptimizer/) | Implements the Coordinate Descent optimization algorithm for numerical optimization problems. |
| [`DFPOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/dfpoptimizer/) | Implements the Davidon-Fletcher-Powell (DFP) optimization algorithm for numerical optimization problems. |
| [`DifferentialEvolutionOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/differentialevolutionoptimizer/) | Implements the Differential Evolution optimization algorithm for numerical optimization problems. |
| [`FTRLOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/ftrloptimizer/) | Represents a Follow The Regularized Leader (FTRL) optimizer for machine learning models. |
| [`GeneticAlgorithmOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/geneticalgorithmoptimizer/) | Represents a Genetic Algorithm optimizer for machine learning models. |
| [`GradientDescentOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/gradientdescentoptimizer/) | Represents a Gradient Descent optimizer for machine learning models. |
| [`LAMBOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/lamboptimizer/) | Implements the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm. |
| [`LARSOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/larsoptimizer/) | Implements the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm. |
| [`LBFGSFunctionOptimizer<T>`](/docs/reference/wiki/optimizers/lbfgsfunctionoptimizer/) | L-BFGS (Limited-memory BFGS) optimizer for minimizing a scalar function of a vector. |
| [`LBFGSOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/lbfgsoptimizer/) | Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm. |
| [`LevenbergMarquardtOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/levenbergmarquardtoptimizer/) | Implements the Levenberg-Marquardt optimization algorithm for non-linear least squares problems. |
| [`LionOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/lionoptimizer/) | Implements the Lion (Evolved Sign Momentum) optimization algorithm for gradient-based optimization. |
| [`MiniBatchGradientDescentOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/minibatchgradientdescentoptimizer/) | Implements the Mini-Batch Gradient Descent optimization algorithm. |
| [`ModifiedGradientDescentOptimizer<T>`](/docs/reference/wiki/optimizers/modifiedgradientdescentoptimizer/) | Modified Gradient Descent optimizer for Hope architecture. |
| [`MomentumOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/momentumoptimizer/) | Implements the Momentum optimization algorithm for gradient-based optimization. |
| [`NadamOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/nadamoptimizer/) | Implements the Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm. |
| [`NelderMeadOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/neldermeadoptimizer/) | Implements the Nelder-Mead optimization algorithm, also known as the downhill simplex method. |
| [`NesterovAcceleratedGradientOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/nesterovacceleratedgradientoptimizer/) | Implements the Nesterov Accelerated Gradient optimization algorithm. |
| [`NewtonMethodOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/newtonmethodoptimizer/) | Implements the Newton's Method optimization algorithm. |
| [`NormalOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/normaloptimizer/) | Implements a normal optimization algorithm with adaptive parameters. |
| [`OptimizationDataBatcher<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/optimizationdatabatcher/) | Provides batch iteration utilities for optimization input data. |
| [`ParticleSwarmOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/particleswarmoptimizer/) | Implements a Particle Swarm Optimization algorithm for finding optimal solutions. |
| [`ProximalGradientDescentOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/proximalgradientdescentoptimizer/) | Implements a Proximal Gradient Descent optimization algorithm which combines gradient descent with regularization. |
| [`RootMeanSquarePropagationOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/rootmeansquarepropagationoptimizer/) | Implements the Root Mean Square Propagation (RMSProp) optimization algorithm, an adaptive learning rate method. |
| [`StochasticGradientDescentOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/stochasticgradientdescentoptimizer/) | Represents a Stochastic Gradient Descent (SGD) optimizer for machine learning models. |
| [`TabuSearchOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/tabusearchoptimizer/) | Represents a Tabu Search optimizer for machine learning models. |
| [`TrustRegionOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/trustregionoptimizer/) | Implements the Trust Region optimization algorithm for machine learning models. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`GradientBasedOptimizerBase<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/gradientbasedoptimizerbase/) | Represents a base class for gradient-based optimization algorithms. |
| [`OptimizerBase<T, TInput, TOutput>`](/docs/reference/wiki/optimizers/optimizerbase/) | Represents the base class for all optimization algorithms, providing common functionality and interfaces. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`OptimizationDataBatcherExtensions`](/docs/reference/wiki/optimizers/optimizationdatabatcherextensions/) | Extension methods for optimization data batching. |


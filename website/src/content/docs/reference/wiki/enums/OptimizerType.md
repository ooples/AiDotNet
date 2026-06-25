---
title: "OptimizerType"
description: "Defines different optimization algorithms used to train machine learning models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines different optimization algorithms used to train machine learning models.

## For Beginners

Optimizers are like the "learning strategy" for AI models. When an AI model is learning, 
it needs to find the best values for its internal settings (called parameters or weights). Optimizers 
are different methods for finding these best values efficiently. Think of them as different strategies 
for climbing a mountain to reach the peak - some take small careful steps, others take bigger leaps, 
and some use special techniques to avoid getting stuck on small hills before reaching the highest peak.

## Fields

| Field | Summary |
|:-----|:--------|
| `AMSGrad` | Variant of Adam that maintains the maximum of past squared gradients for better convergence. |
| `AdaDelta` | Extension of Adagrad that uses a moving average of squared gradients to adapt the learning rate. |
| `AdaGrad` | Adaptive gradient algorithm that accumulates squared gradients to scale the learning rate. |
| `AdaMax` | Variant of Adam that uses the infinity norm for more stable updates. |
| `Adadelta` | Extension of Adagrad that uses a different approach to address the diminishing learning rates problem. |
| `Adagrad` | Adaptive learning rate method that scales learning rates individually for each parameter. |
| `Adam` | Adaptive Moment Estimation - combines the benefits of momentum and RMSProp optimizers. |
| `AdamOptimizer` | Combination of adaptive learning rates and momentum for efficient optimization. |
| `AdamW` | Variant of Adam that implements a decoupled weight decay for better generalization. |
| `Adamax` | Variant of Adam that uses the infinity norm instead of the L2 norm. |
| `AdaptiveGradient` | Adaptive gradient algorithm that maintains per-parameter learning rates. |
| `AntColony` | Nature-inspired algorithm based on the foraging behavior of ant colonies. |
| `BayesianOptimization` | Optimization approach that builds a probabilistic model of the objective function. |
| `ConjugateGradient` | Optimization method that generates search directions that don't interfere with previous progress. |
| `CoordinateDescent` | Optimization method that updates one parameter at a time while keeping others fixed. |
| `CrossEntropy` | Optimization method that generates random samples and selects the best performing ones. |
| `DifferentialEvolution` | Evolutionary algorithm that creates new solutions by combining existing ones. |
| `EvolutionaryAlgorithm` | Evolutionary algorithm that uses principles of natural selection to optimize solutions. |
| `FTRL` | Online learning algorithm that adapts to the data characteristics. |
| `GeneticAlgorithm` | Evolutionary algorithm inspired by natural selection and genetics. |
| `GradientDescent` | Classic optimization algorithm that updates parameters in the direction of steepest descent. |
| `HillClimbing` | Simple optimization strategy that always moves in the direction of immediate improvement. |
| `LBFGS` | Second-order optimization method that approximates the Hessian matrix to accelerate convergence. |
| `Lion` | Optimization algorithm that combines the Lion optimizer with advanced activation techniques. |
| `Momentum` | Gradient descent variant that adds a fraction of the previous update to the current one. |
| `Nadam` | Combination of Adam and Nesterov momentum for improved performance. |
| `NelderMead` | Direct search method that doesn't require gradient information. |
| `NestedLearning` | Nested Learning optimizer - a multi-level optimization paradigm for continual learning. |
| `NesterovAcceleratedGradient` | Advanced variant of Momentum that looks ahead to where the current momentum would take it. |
| `Normal` | Standard optimization approach with fixed learning rate. |
| `NormalOptimizer` | Optimizer that combines random search with adaptive parameter tuning. |
| `ParticleSwarm` | Swarm intelligence algorithm inspired by the social behavior of birds or fish. |
| `PowellMethod` | Optimization method that performs sequential line minimizations along conjugate directions. |
| `QuasiNewton` | Optimization method that approximates second-order information for better search directions. |
| `RAdam` | Variant of Adam that rectifies the variance of the adaptive learning rate. |
| `RMSProp` | Extension of Adagrad that addresses the diminishing learning rates problem. |
| `SimulatedAnnealing` | Probabilistic technique inspired by the annealing process in metallurgy. |
| `StochasticGradientDescent` | Variant of gradient descent that uses random subsets of data for each update. |
| `TrustRegion` | Optimization method that uses a simplified model within a trusted region. |


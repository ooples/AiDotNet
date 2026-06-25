---
title: "Hyperparameter Optimization"
description: "All 20 public types in the AiDotNet.hyperparameteroptimization namespace, organized by kind."
section: "API Reference"
---

**20** public types in this namespace, organized by kind.

## Models & Types (13)

| Type | Summary |
|:-----|:--------|
| [`ASHAOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/hyperparameteroptimization/ashaoptimizer/) | Implements ASHA (Asynchronous Successive Halving Algorithm) for hyperparameter optimization. |
| [`BayesianOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/hyperparameteroptimization/bayesianoptimizer/) | Implements Bayesian optimization for hyperparameter tuning using Gaussian Process regression. |
| [`BracketInfo`](/docs/reference/wiki/hyperparameteroptimization/bracketinfo/) | Information about a Hyperband bracket. |
| [`EarlyStoppingState`](/docs/reference/wiki/hyperparameteroptimization/earlystoppingstate/) | Represents the current state of early stopping. |
| [`EarlyStopping<T>`](/docs/reference/wiki/hyperparameteroptimization/earlystopping/) | Provides early stopping functionality for hyperparameter optimization and training. |
| [`GridSearchOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/hyperparameteroptimization/gridsearchoptimizer/) | Implements grid search hyperparameter optimization. |
| [`HyperbandOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/hyperparameteroptimization/hyperbandoptimizer/) | Implements Hyperband optimization for hyperparameter tuning with early stopping. |
| [`PopulationBasedTrainingOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/hyperparameteroptimization/populationbasedtrainingoptimizer/) | Implements Population-based Training (PBT) for hyperparameter optimization. |
| [`PopulationMemberInfo`](/docs/reference/wiki/hyperparameteroptimization/populationmemberinfo/) | Information about a population member in PBT. |
| [`RandomSearchOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/hyperparameteroptimization/randomsearchoptimizer/) | Implements random search hyperparameter optimization. |
| [`RungStatistics`](/docs/reference/wiki/hyperparameteroptimization/rungstatistics/) | Statistics for a single ASHA rung. |
| [`TrialPrunerStatistics`](/docs/reference/wiki/hyperparameteroptimization/trialprunerstatistics/) | Statistics about trial pruning. |
| [`TrialPruner<T>`](/docs/reference/wiki/hyperparameteroptimization/trialpruner/) | Provides trial pruning functionality for hyperparameter optimization. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`HyperparameterOptimizerBase<T, TInput, TOutput>`](/docs/reference/wiki/hyperparameteroptimization/hyperparameteroptimizerbase/) | Base class for hyperparameter optimization algorithms. |

## Enums (5)

| Type | Summary |
|:-----|:--------|
| [`AcquisitionFunctionType`](/docs/reference/wiki/hyperparameteroptimization/acquisitionfunctiontype/) | Types of acquisition functions for Bayesian optimization. |
| [`EarlyStoppingMode`](/docs/reference/wiki/hyperparameteroptimization/earlystoppingmode/) | Mode for determining improvement in early stopping. |
| [`ExploitStrategy`](/docs/reference/wiki/hyperparameteroptimization/exploitstrategy/) | Strategy for exploiting better performers in PBT. |
| [`ExploreStrategy`](/docs/reference/wiki/hyperparameteroptimization/explorestrategy/) | Strategy for exploring hyperparameters in PBT. |
| [`PruningStrategy`](/docs/reference/wiki/hyperparameteroptimization/pruningstrategy/) | Strategy for pruning trials. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`EarlyStoppingBuilder<T>`](/docs/reference/wiki/hyperparameteroptimization/earlystoppingbuilder/) | Builder for configuring early stopping with fluent API. |


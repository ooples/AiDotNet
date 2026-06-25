---
title: "PopulationBasedTrainingOptimizer<T, TInput, TOutput>"
description: "Implements Population-based Training (PBT) for hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Implements Population-based Training (PBT) for hyperparameter optimization.

## How It Works

**For Beginners:** Population-based Training is an evolutionary approach that:

- Trains a population of models simultaneously
- Periodically checks each model's performance
- Poor performers "exploit" better performers (copy their weights/hyperparameters)
- Copies are then "explored" (slightly mutated hyperparameters)
- This creates an online hyperparameter schedule adapted during training

Key concepts:

- Population: Multiple models training in parallel
- Exploit: Copy weights and hyperparameters from a better performer
- Explore: Perturb hyperparameters to discover better values
- Ready: A model is "ready" to exploit/explore after training for some steps

PBT can discover hyperparameter schedules that vary during training,
something that grid/random/Bayesian search cannot do.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PopulationBasedTrainingOptimizer(Boolean,Int32,Int32,Double,Double,ExploitStrategy,ExploreStrategy,Nullable<Int32>)` | Initializes a new instance of the PopulationBasedTrainingOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExploitAndExplore(HyperparameterSearchSpace)` | Performs the exploit and explore phase. |
| `ExploreConfiguration(Dictionary<String,Object>,HyperparameterSearchSpace)` | Perturbs a configuration's hyperparameters. |
| `GetBestMember` | Gets the best member of the population. |
| `GetPopulationState` | Gets the current population state. |
| `InitializePopulation(HyperparameterSearchSpace)` | Initializes the population with random configurations. |
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Runs Population-based Training optimization. |
| `PerturbParameter(Object,ParameterDistribution)` | Perturbs a single parameter value. |
| `SampleRandomConfiguration(HyperparameterSearchSpace)` | Samples a random configuration from the search space. |
| `SelectExploitTarget(List<PopulationBasedTrainingOptimizer<,,>.PopulationMember>,Int32)` | Selects a member to exploit from. |
| `SelectProbabilistic(List<PopulationBasedTrainingOptimizer<,,>.PopulationMember>,Int32)` | Selects a target using fitness-proportional selection. |
| `SuggestNext(HyperparameterTrial<>)` | Suggests the next hyperparameter configuration to try. |


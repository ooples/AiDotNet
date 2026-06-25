---
title: "AdaptiveParametersHelper<T, TInput, TOutput>"
description: "Helper class that provides methods for dynamically adjusting genetic algorithm parameters during optimization."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Helper class that provides methods for dynamically adjusting genetic algorithm parameters during optimization.

## For Beginners

This helper class contains methods that automatically adjust the settings of a genetic algorithm
while it's running to help it find better solutions.

A genetic algorithm is an AI technique inspired by natural evolution - it creates a "population" of possible 
solutions, selects the best ones, and combines them to create new solutions, similar to how animals evolve 
through natural selection.

Two important settings in genetic algorithms are:

- Crossover rate: How often solutions are combined to create new ones (like breeding)
- Mutation rate: How often random changes are introduced to solutions (like genetic mutations)

This class helps the algorithm perform better by automatically adjusting these rates based on whether
the algorithm is making progress or getting stuck.

## Methods

| Method | Summary |
|:-----|:--------|
| `UpdateAdaptiveGeneticParameters(,,OptimizationStepData<,,>,OptimizationStepData<,,>,GeneticAlgorithmOptimizerOptions<,,>)` | Updates the crossover and mutation rates based on whether the optimization is improving. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides numeric operations appropriate for the generic type T. |


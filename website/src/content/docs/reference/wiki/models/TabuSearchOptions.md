---
title: "TabuSearchOptions<T, TInput, TOutput>"
description: "Configuration options for Tabu Search, a metaheuristic optimization algorithm that enhances local search by using memory structures to avoid revisiting previously explored solutions."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Tabu Search, a metaheuristic optimization algorithm that enhances
local search by using memory structures to avoid revisiting previously explored solutions.

## For Beginners

Tabu Search is like exploring a maze while keeping a list of recently visited paths to avoid.

When solving optimization problems:

- Simple local search methods often get stuck in "local optima" (solutions that are better than their neighbors but not the best overall)
- They can also waste time revisiting the same solutions repeatedly

Tabu Search solves this by:

- Keeping a "tabu list" of recently visited solutions that are temporarily forbidden
- Forcing the search to explore new areas even if they initially seem worse
- Adaptively adjusting its parameters based on search progress
- Combining elements of memory, neighborhood exploration, and strategic oscillation

This approach offers several benefits:

- Effectively escapes local optima
- Avoids cycling through the same solutions
- Efficiently explores the solution space
- Works well for complex combinatorial problems

This class lets you configure how the Tabu Search algorithm behaves.

## How It Works

Tabu Search is an iterative neighborhood search algorithm that enhances local search by avoiding points 
in the search space that have already been visited. The main feature of Tabu Search is the use of explicit 
memory (the tabu list) with two goals: to prevent the search from revisiting previously visited solutions, 
and to explore unvisited areas of the solution space. This approach helps escape local optima and avoid 
cycling in the search process. Tabu Search is particularly effective for combinatorial optimization problems 
where the search space is discrete and contains many local optima. This class inherits from 
GeneticAlgorithmOptimizerOptions and adds parameters specific to Tabu Search, such as tabu list size, 
neighborhood size, and various adaptive parameters for controlling the search process.

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialMutationRate` | Gets or sets the initial mutation rate. |
| `InitialNeighborhoodSize` | Gets or sets the initial size of the neighborhood. |
| `InitialTabuListSize` | Gets or sets the initial size of the tabu list. |
| `MaxFeatureRatio` | Gets or sets the maximum ratio of features to consider in the search. |
| `MaxMutationRate` | Gets or sets the maximum mutation rate. |
| `MaxNeighborhoodSize` | Gets or sets the maximum size of the neighborhood. |
| `MaxTabuListSize` | Gets or sets the maximum size of the tabu list. |
| `MinFeatureRatio` | Gets or sets the minimum ratio of features to consider in the search. |
| `MinMutationRate` | Gets or sets the minimum mutation rate. |
| `MinNeighborhoodSize` | Gets or sets the minimum size of the neighborhood. |
| `MinTabuListSize` | Gets or sets the minimum size of the tabu list. |
| `MutationRate` | Gets or sets the probability of mutation in the search process. |
| `NeighborhoodSize` | Gets or sets the size of the neighborhood to explore in each iteration. |
| `NeighborhoodSizeDecay` | Gets or sets the decay factor for the neighborhood size. |
| `NeighborhoodSizeIncrease` | Gets or sets the increase factor for the neighborhood size. |
| `PerturbationFactor` | Gets or sets the factor for perturbing solutions to generate neighbors. |
| `TabuListSize` | Gets or sets the size of the tabu list. |
| `TabuListSizeDecay` | Gets or sets the decay factor for the tabu list size. |
| `TabuListSizeIncrease` | Gets or sets the increase factor for the tabu list size. |


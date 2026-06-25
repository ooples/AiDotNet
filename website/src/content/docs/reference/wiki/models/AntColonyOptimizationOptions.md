---
title: "AntColonyOptimizationOptions<T, TInput, TOutput>"
description: "Configuration options for the Ant Colony Optimization algorithm, which is inspired by the foraging behavior of ants to find optimal paths through a search space."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Ant Colony Optimization algorithm, which is inspired by the foraging behavior of ants
to find optimal paths through a search space.

## For Beginners

Ant Colony Optimization works like real ants finding food. When ants find food, they leave
a chemical trail (pheromone) on their way back to the nest. Other ants follow strong trails, making them stronger.
Trails that don't lead to food gradually fade away. This algorithm mimics this behavior to solve problems by having
virtual "ants" explore possible solutions and leave "trails" on good solutions that guide future searches.
It's especially good at finding efficient routes or combinations when there are many possibilities to consider.

## How It Works

Ant Colony Optimization (ACO) is a probabilistic technique for solving computational problems that can be reduced to finding
good paths through graphs. It simulates the behavior of ants seeking a path between their colony and a food source.

## Properties

| Property | Summary |
|:-----|:--------|
| `AntCount` | Gets or sets the number of artificial ants used in each iteration of the algorithm. |
| `Beta` | Gets or sets the importance of heuristic information (problem-specific knowledge) relative to pheromone trails. |
| `InitialPheromoneEvaporationRate` | Gets or sets the initial rate at which pheromone trails evaporate over time. |
| `InitialPheromoneIntensity` | Gets or sets the initial strength of pheromone deposits left by ants. |
| `MaxPheromoneEvaporationRate` | Gets or sets the maximum allowed value for the pheromone evaporation rate. |
| `MaxPheromoneIntensity` | Gets or sets the maximum allowed value for pheromone intensity on any path. |
| `MinPheromoneEvaporationRate` | Gets or sets the minimum allowed value for the pheromone evaporation rate. |
| `MinPheromoneIntensity` | Gets or sets the minimum allowed value for pheromone intensity on any path. |
| `PheromoneEvaporationRateDecay` | Gets or sets the factor by which to decrease the pheromone evaporation rate when adaptation is needed. |
| `PheromoneEvaporationRateIncrease` | Gets or sets the factor by which to increase the pheromone evaporation rate when adaptation is needed. |
| `PheromoneIntensityDecay` | Gets or sets the factor by which to decrease the pheromone intensity when adaptation is needed. |
| `PheromoneIntensityIncrease` | Gets or sets the factor by which to increase the pheromone intensity when adaptation is needed. |


---
title: "SimulatedAnnealingOptions<T, TInput, TOutput>"
description: "Configuration options for the Simulated Annealing optimization algorithm, a probabilistic technique for approximating the global optimum of a given function."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Simulated Annealing optimization algorithm, a probabilistic technique
for approximating the global optimum of a given function.

## For Beginners

Simulated Annealing is an optimization technique inspired by metallurgy.

When metalworkers want to remove defects from metals, they:

- Heat the metal to a high temperature (atoms move freely)
- Slowly cool it down (atoms gradually settle into a low-energy state)

Simulated Annealing works similarly to find the best solution to a problem:

- It starts with a high "temperature" where it makes big, sometimes random changes
- It gradually "cools down," making smaller, more careful adjustments
- This approach helps it avoid getting stuck in mediocre solutions

The key insight is that sometimes accepting a worse solution temporarily
helps you find a better solution in the long run:

- At high temperatures, it frequently accepts worse solutions to explore widely
- At low temperatures, it rarely accepts worse solutions, focusing on refinement

This is particularly useful for complex problems with many possible solutions,
like route optimization, scheduling, or parameter tuning.

This class lets you configure exactly how the algorithm heats, cools, and explores
the solution space.

## How It Works

Simulated Annealing is a probabilistic optimization algorithm inspired by the annealing process in metallurgy, 
where metals are heated and then slowly cooled to reduce defects. In optimization, it uses a similar approach 
to find the global minimum of a function by occasionally accepting worse solutions to escape local minima. 
The algorithm starts with a high "temperature" that allows for large random moves in the solution space, 
including accepting worse solutions with high probability. As the temperature decreases according to a cooling 
schedule, the algorithm becomes more selective, gradually focusing on promising regions of the solution space. 
This class provides configuration options for controlling the annealing process, including temperature parameters, 
iteration limits, and neighborhood generation settings. Simulated Annealing is particularly useful for complex 
optimization problems with many local optima where deterministic methods might get trapped.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimulatedAnnealingOptions` | Initializes a new instance of the SimulatedAnnealingOptions class with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CoolingRate` | Gets or sets the cooling rate for the temperature reduction schedule. |
| `InitialTemperature` | Gets or sets the initial temperature of the annealing process. |
| `MaxIterationsWithoutImprovement` | Gets or sets the maximum number of consecutive iterations without improvement before early stopping. |
| `MaxNeighborGenerationRange` | Gets or sets the maximum range for generating neighboring solutions. |
| `MaxTemperature` | Gets or sets the maximum temperature allowed during the annealing process. |
| `MinNeighborGenerationRange` | Gets or sets the minimum range for generating neighboring solutions. |
| `MinTemperature` | Gets or sets the minimum temperature at which the annealing process stops. |
| `NeighborGenerationRange` | Gets or sets the initial range for generating neighboring solutions. |


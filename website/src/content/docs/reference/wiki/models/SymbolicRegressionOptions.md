---
title: "SymbolicRegressionOptions"
description: "Configuration options for Symbolic Regression, an evolutionary approach to finding mathematical expressions that best fit a dataset."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Symbolic Regression, an evolutionary approach to finding
mathematical expressions that best fit a dataset.

## For Beginners

Symbolic Regression finds mathematical formulas that explain your data.

When performing regression (predicting values):

- Traditional methods fit parameters to a predefined equation
- You must specify the form of the equation in advance
- This requires knowing what relationship to look for

Symbolic Regression solves this by:

- Automatically discovering both the structure and parameters of equations
- Starting with a population of random simple formulas
- Evolving them through "survival of the fittest"
- Combining good formulas to create better ones (crossover)
- Randomly changing formulas occasionally (mutation)
- Continuing until it finds a formula that fits the data well

This approach offers several benefits:

- Can discover unexpected relationships in your data
- Produces human-readable mathematical formulas
- Doesn't require prior knowledge of the underlying relationship
- Often finds simpler models than other techniques

This class lets you configure how the evolutionary algorithm searches for formulas.

## How It Works

Symbolic Regression is a type of regression analysis that searches for mathematical expressions that best 
fit a given dataset, both in terms of accuracy and simplicity. Unlike traditional regression techniques 
that fit parameters to a predefined model structure, symbolic regression simultaneously evolves both the 
structure of the model and its parameters. It uses genetic programming, an evolutionary algorithm inspired 
by biological evolution, to evolve a population of mathematical expressions through operations like 
selection, crossover, and mutation. This approach can discover complex, non-linear relationships in data 
without requiring prior assumptions about the form of the model. This class inherits from 
NonLinearRegressionOptions and adds parameters specific to the evolutionary algorithm used in symbolic 
regression, such as population size, number of generations, and genetic operator rates.

## Properties

| Property | Summary |
|:-----|:--------|
| `CrossoverRate` | Gets or sets the probability of crossover in the genetic algorithm. |
| `FitnessThreshold` | Gets or sets the fitness threshold for early stopping. |
| `MaxGenerations` | Gets or sets the maximum number of generations for the genetic algorithm. |
| `MutationRate` | Gets or sets the probability of mutation in the genetic algorithm. |
| `PopulationSize` | Gets or sets the size of the population in the genetic algorithm. |


---
title: "IGeneticAlgorithm<T, TInput, TOutput, TIndividual, TGene>"
description: "Represents a machine learning model that uses genetic algorithms or evolutionary computation while maintaining the core capabilities of a full model."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a machine learning model that uses genetic algorithms or evolutionary computation
while maintaining the core capabilities of a full model.

## For Beginners

This interface provides functionality for AI models that use genetic algorithms - methods
inspired by natural evolution.

Think of a genetic model as a population of potential solutions that evolve over time:

1. Population Management
- The model maintains multiple candidate solutions (individuals)
- Each individual represents a possible solution to your problem

2. Evolution Process
- Individuals are evaluated based on how well they solve the problem (fitness)
- The best individuals are selected to "reproduce" (selection)
- New individuals are created by combining parts of successful ones (crossover)
- Random changes are introduced to maintain diversity (mutation)
- This process repeats over many generations, with solutions improving over time

3. Advantages
- Can solve complex problems where traditional algorithms struggle
- Often finds creative solutions humans might not consider
- Good for optimization problems and symbolic regression
- Can adapt to changing conditions and problems

This is particularly useful for problems like:

- Finding optimal neural network architectures
- Symbolic regression (discovering mathematical equations from data)
- Optimizing complex systems with many parameters
- Evolving game-playing strategies or agent behaviors

## How It Works

This interface extends the IFullModel interface by adding genetic algorithm capabilities
that allow for evolutionary optimization, including population management, crossover operations,
mutation, selection, and fitness evaluation through integration with IFitnessCalculator.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCrossoverOperator(String,Func<,,Double,ICollection<>>)` | Adds a custom crossover operator. |
| `AddMutationOperator(String,Func<,Double,>)` | Adds a custom mutation operator. |
| `ConfigureGeneticParameters(GeneticParameters)` | Configures the genetic algorithm parameters. |
| `CreateIndividual(ICollection<>)` | Creates a new individual with the specified genes. |
| `Crossover(,,Double)` | Performs crossover between two parent individuals to produce offspring. |
| `EvaluateIndividual(,,,,)` | Evaluates an individual by converting it to a model and generating evaluation data. |
| `Evolve(Int32,,,,,Func<EvolutionStats<,,>,Boolean>)` | Evolves the population for a specified number of generations. |
| `GetBestIndividual` | Gets the best individual from the current population. |
| `GetEvolutionStats(IFitnessCalculator<,,>)` | Gets statistics about the current evolutionary state, including generation number, population diversity, and fitness distribution. |
| `GetFitnessCalculator` | Gets the fitness calculator used to evaluate individuals. |
| `GetGeneticParameters` | Gets the current genetic algorithm parameters. |
| `GetPopulation` | Gets the current population of individuals in the genetic model. |
| `IndividualToModel()` | Converts an individual to a trained model that can make predictions. |
| `InitializePopulation(Int32,InitializationMethod)` | Initializes a new population with random individuals. |
| `LoadPopulation(String)` | Loads a population from a file. |
| `Mutate(,Double)` | Applies mutation to an individual. |
| `SavePopulation(String)` | Saves the current population to a file. |
| `Select(Int32,SelectionMethod)` | Selects individuals from the population for reproduction. |
| `SetFitnessCalculator(IFitnessCalculator<,,>)` | Sets the fitness calculator to be used for evaluating individuals. |


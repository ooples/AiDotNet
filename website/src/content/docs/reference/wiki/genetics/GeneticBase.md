---
title: "GeneticBase<T, TInput, TOutput>"
description: "Provides a base implementation of IGeneticModel that handles common genetic algorithm operations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Genetics`

Provides a base implementation of IGeneticModel that handles common genetic algorithm operations.

## For Beginners

This class provides a ready-to-use foundation for genetic algorithm models.
It handles:

- Managing a population of candidate solutions
- Evolving the population through selection, crossover, and mutation
- Tracking statistics about the evolutionary process
- Saving and loading populations

When creating your own genetic model, you can inherit from this class and focus
on the specific implementation details of your model type rather than reimplementing
the entire genetic algorithm framework.

## How It Works

This abstract base class implements the IGeneticModel interface, providing standard
implementations for common genetic algorithm operations while allowing derived classes
to customize behavior specific to their genetic model type.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GeneticBase(IFitnessCalculator<,,>)` | Initializes a new instance of the GeneticModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestIndividual` | The best individual found so far. |
| `CrossoverOperators` | A dictionary mapping crossover operator names to implementations. |
| `CurrentStats` | The current evolution statistics. |
| `EvolutionStopwatch` | A stopwatch for tracking evolution time. |
| `FitnessCalculator` | The fitness calculator used to evaluate individuals. |
| `GeneticParams` | The parameters for the genetic algorithm. |
| `MutationOperators` | A dictionary mapping mutation operator names to implementations. |
| `Population` | The current population of individuals. |
| `Random` | The random number generator used for stochastic operations. |
| `TrainingInputForInitialization` | The training input data, stored before population initialization so derived classes can use it to determine proper parameter dimensions when models have empty parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCrossoverOperator(String,Func<ModelIndividual<,,,ModelParameterGene<>>,ModelIndividual<,,,ModelParameterGene<>>,Double,ICollection<ModelIndividual<,,,ModelParameterGene<>>>>)` | Adds a custom crossover operator. |
| `AddDefaultCrossoverOperators` | Adds default crossover operators. |
| `AddDefaultMutationOperators` | Adds default mutation operators. |
| `AddMutationOperator(String,Func<ModelIndividual<,,,ModelParameterGene<>>,Double,ModelIndividual<,,,ModelParameterGene<>>>)` | Adds a custom mutation operator. |
| `CalculateDiversity` | Calculates the genetic diversity of the population. |
| `CalculateGeneticDistance(ModelIndividual<,,,ModelParameterGene<>>,ModelIndividual<,,,ModelParameterGene<>>)` | Calculates the genetic distance between two individuals. |
| `ConfigureGeneticParameters(GeneticParameters)` | Configures the genetic algorithm parameters. |
| `CreateIndividual(ICollection<ModelParameterGene<>>)` | Creates a new individual with the specified genes. |
| `CreateNextGeneration(,,,)` | Creates the next generation of individuals through selection, crossover, and mutation. |
| `Crossover(ModelIndividual<,,,ModelParameterGene<>>,ModelIndividual<,,,ModelParameterGene<>>,Double)` | Performs crossover between two parent individuals to produce offspring. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `DeserializeIndividual(Byte[])` | Deserializes an individual from a byte array. |
| `DeserializeModelData(Byte[])` | Deserializes model-specific data. |
| `DeserializePopulation(Byte[])` | Deserializes a population from a byte array. |
| `EvaluateIndividual(ModelIndividual<,,,ModelParameterGene<>>,,,,)` | Evaluates an individual by converting it to a model and generating evaluation data. |
| `EvaluateModelForGenetics(ModelEvaluationInput<,,>)` | Evaluates a model for genetic algorithm fitness calculation. |
| `EvaluatePopulation(,,,)` | Evaluates all individuals in the population. |
| `Evolve(Int32,,,,,Func<EvolutionStats<,,>,Boolean>)` | Evolves the population for a specified number of generations. |
| `FindBestIndividual` | Finds the best individual in the current population. |
| `GetBestIndividual` | Gets the best individual from the current population. |
| `GetElites(Int32)` | Gets the elite individuals (best performers) from the population. |
| `GetEvolutionStats(IFitnessCalculator<,,>)` | Gets statistics about the current evolutionary state. |
| `GetFitnessCalculator` | Gets the fitness calculator used to evaluate individuals. |
| `GetGeneticParameters` | Gets the current genetic algorithm parameters. |
| `GetMetaData` | Gets the metadata for the model. |
| `GetPopulation` | Gets the current population of individuals in the genetic model. |
| `IndividualToModel(ModelIndividual<,,,ModelParameterGene<>>)` | Converts an individual to a trained model that can make predictions. |
| `InitializePopulation(Int32,InitializationMethod)` | Initializes a new population with random individuals. |
| `InvertFitness()` | Inverts a fitness score for use in comparisons. |
| `IsBetterFitness(,)` | Determines if one fitness score is better than another. |
| `LoadPopulation(String)` | Loads a population from a file. |
| `Mutate(ModelIndividual<,,,ModelParameterGene<>>,Double)` | Applies mutation to an individual. |
| `MutateGene(ModelParameterGene<>)` | Creates a mutated version of a gene. |
| `MutateGeneGaussian(ModelParameterGene<>)` | Creates a mutated version of a gene using Gaussian noise. |
| `Predict()` | Makes a prediction using the current best model. |
| `RankSelection(Int32)` | Selects individuals using rank selection. |
| `RouletteWheelSelection(Int32)` | Selects individuals using roulette wheel selection. |
| `SavePopulation(String)` | Saves the current population to a file. |
| `Select(Int32,SelectionMethod)` | Selects individuals from the population for reproduction. |
| `Serialize` | Serializes the model to a byte array. |
| `SerializeIndividual(ModelIndividual<,,,ModelParameterGene<>>)` | Serializes an individual to a byte array. |
| `SerializeModelData` | Serializes model-specific data. |
| `SerializePopulation` | Serializes the population to a byte array. |
| `SetFitnessCalculator(IFitnessCalculator<,,>)` | Sets the fitness calculator to be used for evaluating individuals. |
| `StochasticUniversalSamplingSelection(Int32)` | Selects individuals using stochastic universal sampling. |
| `TournamentSelection(Int32)` | Selects individuals using tournament selection. |
| `TruncationSelection(Int32)` | Selects individuals using truncation selection. |
| `UniformSelection(Int32)` | Selects individuals using uniform selection (all individuals have equal probability). |
| `UpdateEvolutionStats` | Updates the evolution statistics based on the current population. |


---
title: "Genetics"
description: "All 17 public types in the AiDotNet.genetics namespace, organized by kind."
section: "API Reference"
---

**17** public types in this namespace, organized by kind.

## Models & Types (16)

| Type | Summary |
|:-----|:--------|
| [`AdaptiveGeneticAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/genetics/adaptivegeneticalgorithm/) |  |
| [`BinaryGene`](/docs/reference/wiki/genetics/binarygene/) | Represents a gene that holds a binary value (0 or 1). |
| [`BinaryIndividual`](/docs/reference/wiki/genetics/binaryindividual/) | Represents an individual encoded with binary genes, suitable for classic GA problems. |
| [`IslandModelGeneticAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/genetics/islandmodelgeneticalgorithm/) |  |
| [`ModelIndividual<T, TInput, TOutput, TGene>`](/docs/reference/wiki/genetics/modelindividual/) | Represents an individual that is also a full model, allowing direct evolution of models without conversion between individuals and models. |
| [`ModelParameterGene<T>`](/docs/reference/wiki/genetics/modelparametergene/) | Represents a gene that corresponds to a parameter in a machine learning model. |
| [`MultiObjectiveRealIndividual`](/docs/reference/wiki/genetics/multiobjectiverealindividual/) | A real-valued individual supporting multi-objective optimization. |
| [`NSGAII<T, TInput, TOutput>`](/docs/reference/wiki/genetics/nsgaii/) |  |
| [`NodeGene`](/docs/reference/wiki/genetics/nodegene/) | Represents a node in a genetic programming tree. |
| [`PermutationGene`](/docs/reference/wiki/genetics/permutationgene/) | Represents a gene in a permutation (the index of an element in a sequence). |
| [`PermutationIndividual`](/docs/reference/wiki/genetics/permutationindividual/) | Represents an individual encoded as a permutation, suitable for problems like TSP. |
| [`RealGene`](/docs/reference/wiki/genetics/realgene/) | Represents a gene with a real (double) value. |
| [`RealValuedIndividual`](/docs/reference/wiki/genetics/realvaluedindividual/) | Represents an individual encoded with real-valued genes, suitable for numerical optimization problems. |
| [`StandardGeneticAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/genetics/standardgeneticalgorithm/) |  |
| [`SteadyStateGeneticAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/genetics/steadystategeneticalgorithm/) |  |
| [`TreeIndividual`](/docs/reference/wiki/genetics/treeindividual/) | Represents an individual in genetic programming with a tree structure. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`GeneticBase<T, TInput, TOutput>`](/docs/reference/wiki/genetics/geneticbase/) | Provides a base implementation of IGeneticModel that handles common genetic algorithm operations. |


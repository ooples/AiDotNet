---
title: "NEAT<T>"
description: "Represents a NeuroEvolution of Augmenting Topologies (NEAT) algorithm implementation, which evolves neural networks through genetic algorithms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a NeuroEvolution of Augmenting Topologies (NEAT) algorithm implementation, which evolves
neural networks through genetic algorithms.

## For Beginners

NEAT is a way to grow neural networks through evolution rather than training them with fixed structures.

Think of NEAT like breeding plants to get better features:

- Instead of designing a neural network by hand, you start with simple networks
- These networks "reproduce" and "mutate" over generations
- Networks that perform better on your task are more likely to pass on their "genes"
- Over time, the networks evolve complex structures that solve your problem well

The key differences from traditional neural networks:

- The structure (connections between neurons) evolves along with the weights
- Networks can grow more complex over time by adding new neurons and connections
- You work with a population of many networks, not just one
- Instead of training with gradient descent, you use evolution to improve performance

NEAT is particularly good for:

- Problems where you don't know the ideal network structure
- Reinforcement learning tasks (like game playing)
- Finding novel solutions that a human designer might not think of

## How It Works

NEAT is an evolutionary algorithm that creates and evolves neural network topologies along with connection weights.
Unlike traditional neural networks with fixed structures, NEAT starts with simple networks and gradually adds
complexity through evolution. It uses genetic operators like mutation and crossover, along with speciation
to protect innovation, to evolve networks that solve specific problems without requiring manual design
of the network architecture.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NEAT` | Initializes a new instance with default settings. |
| `NEAT(NeuralNetworkArchitecture<>,Int32,Double,Double,ILossFunction<>,NEATOptions)` | Initializes a new instance of the `NEAT` class with the specified architecture and evolution parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters (connections) in the best genome. |
| `_crossoverRate` | Gets or sets the probability of crossover occurring during reproduction. |
| `_mutationRate` | Gets or sets the probability of mutation occurring during reproduction. |
| `_populationSize` | Gets or sets the size of the population (number of genomes). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ActivateGenome(Genome<>,Vector<>)` | Activates a genome's neural network with the given input. |
| `ApplySigmoid()` | Applies the sigmoid activation function to a value. |
| `ComputeTopologySignature(List<Connection<>>)` | O(N) FNV-1a hash over every connection slot's `(FromNode, ToNode, IsEnabled)` tuple in iteration order. |
| `CreateInitialGenome` | Creates a single initial genome with connections from each input to each output. |
| `CreateNewInstance` | Creates a new instance of the NEAT model with the same architecture and evolutionary parameters. |
| `Crossover(Genome<>,Genome<>)` | Creates a new genome by combining genetic material from two parent genomes. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes NEAT-specific data from a binary reader. |
| `EvolvePopulation(Func<Genome<>,>,Int32)` | Evolves the population over a specified number of generations using the provided fitness function. |
| `ExtractTrainingData(Tensor<>,Tensor<>)` | Extracts training data pairs from input and expected output tensors. |
| `GetBestGenome` | Gets the genome with the highest fitness from the population. |
| `GetModelMetadata` | Gets metadata about the NEAT model. |
| `GetNamedLayerActivations(Tensor<>)` | Gets named activations from the best genome's network when processing input. |
| `GetOptions` |  |
| `GetOrBuildMaxNodeId(Genome<>,Int32)` | Issue #1392 perf helper: returns max(FromNode, ToNode, biasNodeId) across the genome's enabled connections. |
| `GetOrBuildReferencedNonInputNodeIds(Genome<>,Int32)` | Issue #1392 perf helper: caches the list of node IDs >= InputSize that the sigmoid sweep should touch. |
| `GetOrBuildSortedConnections(Genome<>)` | Issue #1392 perf helper: returns the cached topologically-sorted connection list for `genome`, rebuilding only when the topology signature changed since the last call. |
| `GetParameterChunks` | Yields the best genome's connection weights as a single chunk so snapshot-based parameter-change probes (Training_ShouldChangeParameters, GradientFlow_ShouldBeNonZeroAndFinite) see real evolutionary updates. |
| `GetParameters` | Gets the parameters (connection weights) of the best genome. |
| `InitializeLayers` | Initializes the layers of the neural network. |
| `InitializePopulation` | Creates the initial population of genomes with minimal network structures. |
| `IsReadyToPredict` | Checks if the NEAT model is ready to make predictions. |
| `Mutate(Genome<>)` | Applies random mutations to a genome based on the mutation rate. |
| `PredictCore(Tensor<>)` | Predicts output values for input data using the best genome in the population. |
| `RandomWeight` | Generates a random weight value for neural network connections. |
| `SelectParent` | Selects a parent genome for reproduction using tournament selection. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes NEAT-specific data to a binary writer. |
| `SortConnectionsTopologically(Genome<>)` | Sorts connections in topological order for proper feed-forward activation. |
| `Train(Tensor<>,Tensor<>)` | Trains the NEAT system using supervised learning data. |
| `UpdateParameters(Vector<>)` | Updates the connection weights of the best genome using the provided parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_innovationNumber` | Gets or sets the global innovation number counter used to track historical origins of genes. |
| `_population` | Gets the current population of genomes (neural network structures). |
| `_rng` | Per-instance deterministic RNG for the evolutionary search (selection, crossover, mutation). |


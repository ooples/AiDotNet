---
title: "Genome<T>"
description: "Represents a genome in a neuroevolutionary algorithm, containing a collection of connections between nodes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a genome in a neuroevolutionary algorithm, containing a collection of connections between nodes.

## For Beginners

A Genome is like a blueprint for a neural network.

Think of a Genome as:

- A DNA-like structure that defines how a neural network is built
- A collection of connections (wires) between nodes (neurons)
- Each connection has a weight (strength) and can be enabled or disabled
- Instead of training this network with examples, it evolves through generations

Just as biological organisms evolve through natural selection, these neural network blueprints
can evolve to solve problems through a process of selection, mutation, and reproduction.
The best-performing blueprints are selected to create the next generation.

## How It Works

A Genome is a fundamental data structure in neuroevolutionary algorithms like NEAT (NeuroEvolution of Augmenting Topologies).
It encodes the structure and weights of a neural network as a set of connections between nodes. Each connection has a weight,
an enabled/disabled state, and an innovation number that tracks its evolutionary history. Genomes can be mutated, crossed over,
and evaluated for fitness, allowing neural networks to evolve over generations rather than being trained through traditional
gradient-based methods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Genome(Int32,Int32)` | Initializes a new instance of the `Genome` class with the specified network dimensions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Connections` | Gets the list of connections that make up this genome. |
| `Fitness` | Gets or sets the fitness score of this genome. |
| `InputSize` | Gets the number of input nodes in the neural network. |
| `NumOps` | Gets the numeric operations helper for the specified type T. |
| `OutputSize` | Gets the number of output nodes in the neural network. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Vector<>)` | Activates the neural network encoded by this genome with the given input vector. |
| `AddConnection(Int32,Int32,,Boolean,Int32)` | Adds a new connection to this genome. |
| `Clone` | Creates a deep copy of this genome. |
| `Deserialize(BinaryReader)` | Deserializes this genome from a binary stream. |
| `DisableConnection(Int32)` | Disables a connection with the specified innovation number. |
| `Serialize(BinaryWriter)` | Serializes this genome to a binary stream. |
| `TopologicalSort(List<Connection<>>)` | Sorts the connections in topological order from inputs to outputs. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CachedMaxNodeId` | Issue #1392 perf: max node id referenced by any (FromNode, ToNode) in `Connections` plus the bias-node id, cached alongside the topology signature so `Vector{` can size its flat-array activations buffer in one shot instead of growing a Dict… |
| `CachedTopologySignatureCount` | Issue #1392 perf: per-genome cache for `NEAT.SortConnectionsTopologically` + the "non-input node" set `NEAT.ActivateGenome` walks for activation. |


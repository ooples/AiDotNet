---
title: "Connection<T>"
description: "Represents a connection between two nodes in a neural network, particularly used in evolving neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a connection between two nodes in a neural network, particularly used in evolving neural networks.

## For Beginners

A Connection is like a wire connecting two parts of a neural network.

Think of a neural network as a system of connected nodes (like neurons in a brain):

- Each Connection is like a wire that passes signals from one node to another
- The Weight determines how strong the signal is (like a volume knob)
- IsEnabled acts like an on/off switch for the connection
- The Innovation number is like a birth certificate that shows when this connection first appeared

For example, if node 3 connects to node 5 with a weight of 0.7, signals from node 3 will reach node 5,
but their strength will be multiplied by 0.7 (either amplified or reduced depending on the original value).

## How It Works

A Connection represents a weighted link between two nodes in a neural network. Connections are fundamental
elements in neural networks, allowing signals to flow from one node to another with a specific weight.
This class is particularly designed for use in evolutionary algorithms like NEAT (NeuroEvolution of Augmenting 
Topologies), which evolve both the weights and structure of neural networks. The Innovation number serves as 
a historical marker to track the evolutionary lineage of connections.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Connection(Int32,Int32,,Boolean,Int32)` | Initializes a new instance of the `Connection` class with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FromNode` | Gets the identifier of the node from which the connection originates. |
| `Innovation` | Gets the innovation number, a historical marker that uniquely identifies the connection in the context of evolution. |
| `IsEnabled` | Gets or sets a value indicating whether the connection is enabled and actively transmitting signals. |
| `ToNode` | Gets the identifier of the node to which the connection leads. |
| `Weight` | Gets or sets the weight of the connection, which determines the strength of the signal transmission. |


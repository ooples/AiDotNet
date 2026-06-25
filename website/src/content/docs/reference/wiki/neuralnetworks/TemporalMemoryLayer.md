---
title: "TemporalMemoryLayer<T>"
description: "Represents a temporal memory layer that models sequence learning through hierarchical temporal memory concepts."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a temporal memory layer that models sequence learning through hierarchical temporal memory concepts.

## For Beginners

This layer helps the network remember and predict sequences of patterns.

Think of it like learning to anticipate what comes next in a song:

- The layer organizes memory cells into columns (like musical notes)
- Each column can have multiple cells (representing different contexts for the same note)
- When a note plays, the layer activates specific cells based on what came before
- Over time, it learns which notes typically follow others in different contexts

For example, in a melody, the note "C" might be followed by "D" in the verse but by "G" in the chorus.
This layer helps the network learn such context-dependent sequences by remembering not just what
happened, but the context in which it happened.

## How It Works

A temporal memory layer implements a simplified version of Hierarchical Temporal Memory (HTM) concepts to learn
sequential patterns in data. It organizes cells into columns, where cells within the same column represent
alternative contexts for the same input pattern. This allows the layer to maintain multiple predictions
simultaneously and learn temporal patterns in the input data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalMemoryLayer(Int32,Int32)` | Initializes a new instance of the `TemporalMemoryLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `PreviousState` | Gets or sets the previous input state of the layer. |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the temporal memory layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameters` | Gets all cell states of the layer as a single vector. |
| `GetPredictions` | Gets the predicted columns based on the current cell states. |
| `InitializeCellStates` | Initializes all cell states to zero. |
| `Learn(Vector<>,Vector<>)` | Updates the cell states based on the current input and previous state. |
| `ResetState` | Resets the internal state of the layer. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `UpdateParameters()` | Updates the parameters of the layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CellStates` | The states of all cells in the temporal memory layer. |
| `CellsPerColumn` | The number of cells per column in the temporal memory layer. |
| `ColumnCount` | The number of columns in the temporal memory layer. |


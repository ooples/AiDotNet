---
title: "Checkpoint<T, TInput, TOutput>"
description: "Represents a saved checkpoint of model training state."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a saved checkpoint of model training state.

## How It Works

**For Beginners:** A checkpoint is like a save point in a video game - it captures
everything needed to resume training from that exact point.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Checkpoint` | Initializes a new instance of the Checkpoint class. |
| `Checkpoint(Object,Dictionary<String,Object>,String,Int32,Int32,Dictionary<String,>,Dictionary<String,Object>)` | Initializes a new instance of the Checkpoint class with specified values. |
| `Checkpoint(Object,IOptimizer<,,>,Int32,Int32,Dictionary<String,>,Dictionary<String,Object>)` | Initializes a new instance of the Checkpoint class with an optimizer object. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointId` | Gets the unique identifier for this checkpoint. |
| `CreatedAt` | Gets the timestamp when this checkpoint was created. |
| `Epoch` | Gets or sets the training epoch number. |
| `FilePath` | Gets or sets the file path where the checkpoint is stored. |
| `Metadata` | Gets or sets additional metadata. |
| `Metrics` | Gets or sets the performance metrics at this checkpoint. |
| `Model` | Gets or sets the model at this checkpoint. |
| `OptimizerState` | Gets or sets the optimizer state as a serializable dictionary. |
| `OptimizerTypeName` | Gets or sets the optimizer type name for reconstruction. |
| `Step` | Gets or sets the training step number. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractOptimizerState(IOptimizer<,,>)` | Extracts serializable state from an optimizer. |


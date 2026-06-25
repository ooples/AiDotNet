---
title: "ModelIndividual<T, TInput, TOutput, TGene>"
description: "Represents an individual that is also a full model, allowing direct evolution of models without conversion between individuals and models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Genetics`

Represents an individual that is also a full model, allowing direct evolution of models
without conversion between individuals and models.

## For Beginners

This class combines the functionality of an individual in a genetic algorithm with a machine
learning model. This means:

- You can evolve the model directly without converting between different representations
- The individual can make predictions like any other model
- It simplifies the implementation of genetic algorithms for model optimization

Use this when you want to directly evolve machine learning models using genetic algorithms.

## How It Works

This class implements both IEvolvable and IFullModel interfaces, allowing it to be used
directly in genetic algorithms while also providing model prediction capabilities.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelIndividual(ICollection<>,Func<ICollection<>,IFullModel<,,>>)` | Creates a new model individual with the specified genes and model factory. |
| `ModelIndividual(IFullModel<,,>,ICollection<>,Func<ICollection<>,IFullModel<,,>>)` | Creates a new model individual by wrapping an existing model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function for gradient computation by delegating to the inner model. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients by delegating to the inner model. |
| `Clone` | Creates a deep clone of this individual. |
| `ComputeGradients(,,ILossFunction<>)` | Computes gradients by delegating to the inner model. |
| `Deserialize(Byte[])` | Deserializes the model from a byte array. |
| `Dispose` |  |
| `Dispose(Boolean)` | Disposes the inner model. |
| `GetFitness` | Gets the fitness of this individual. |
| `GetGenes` | Gets the genes of this individual. |
| `GetMetaData` | Gets the metadata for the model. |
| `GetParameters` | Gets the parameters of the model. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `Predict()` | Makes a prediction using the inner model. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` | Serializes the model to a byte array. |
| `SetFitness()` | Sets the fitness of this individual. |
| `SetGenes(ICollection<>)` | Sets the genes of this individual and updates the inner model. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the model. |
| `WithParameters(Vector<>)` | Creates a new model with the specified parameters. |


---
title: "OptimizerFactory<T, TInput, TOutput>"
description: "A factory class that creates optimizer instances for training machine learning models."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Factories`

A factory class that creates optimizer instances for training machine learning models.

## For Beginners

An optimizer is an algorithm that adjusts the parameters of a machine learning model 
to minimize errors and improve performance. Think of it like a navigator that helps your model find the 
best path to the correct answers.

## How It Works

This factory helps you create different types of optimizers without needing to know their internal 
implementation details. Think of it like ordering a specific tool from a catalog - you just specify 
what you need, and the factory provides it.

## Methods

| Method | Summary |
|:-----|:--------|
| `#cctor` | Static constructor that initializes the optimizer type dictionary. |
| `CreateOptimizer(OptimizerType)` | Creates an optimizer of the specified type with default options and no model. |
| `GetOptimizerType(IOptimizer<,,>)` | Determines the optimizer type from an existing optimizer instance. |
| `RegisterOptimizerType(OptimizerType,Type)` | Registers an optimizer type with its corresponding implementation class. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_optimizerTypes` | A dictionary that maps optimizer types to their corresponding implementation classes. |


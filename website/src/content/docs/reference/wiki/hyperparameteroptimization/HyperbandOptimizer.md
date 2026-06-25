---
title: "HyperbandOptimizer<T, TInput, TOutput>"
description: "Implements Hyperband optimization for hyperparameter tuning with early stopping."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Implements Hyperband optimization for hyperparameter tuning with early stopping.

## How It Works

**For Beginners:** Hyperband is a smart resource allocation strategy that:

- Trains many configurations with minimal resources initially
- Progressively eliminates poorly performing configurations
- Allocates more resources to promising configurations
- Uses "successive halving" to efficiently explore the search space

Key concepts:

- Resource (R): Training budget (e.g., epochs, iterations, data samples)
- Configuration: A specific set of hyperparameter values
- Bracket: A group of configurations competing via successive halving
- Successive Halving: Repeatedly train, evaluate, and keep top half

Hyperband runs multiple brackets with different exploration/exploitation trade-offs,
combining the best properties of random search with aggressive early stopping.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperbandOptimizer(Boolean,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the HyperbandOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumBrackets` | Gets the number of brackets in this Hyperband configuration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBracketInfo` | Gets detailed information about the bracket structure. |
| `GetTotalConfigurationCount` | Gets the total number of configurations that would be evaluated in a full Hyperband run. |
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameter configuration using Hyperband. |
| `RunBracket(Int32,Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32,Int32)` | Runs a single Hyperband bracket with successive halving. |
| `SampleRandomConfiguration(HyperparameterSearchSpace)` | Samples a random configuration from the search space. |
| `SuggestNext(HyperparameterTrial<>)` | Suggests the next hyperparameter configuration to try. |


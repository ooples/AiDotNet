---
title: "LossFunctionFactory<T>"
description: "Factory for creating loss function instances from `LossType` enum values."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Training.Factories`

Factory for creating loss function instances from `LossType` enum values.

## For Beginners

This factory creates loss functions based on a simple name or enum value.
You don't need to know the exact class name or constructor details - just specify the type
and optional parameters.

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(LossType)` | Creates a loss function of the specified type with default parameters. |
| `Create(LossType,Dictionary<String,Object>)` | Creates a loss function of the specified type with optional parameters. |
| `Create(String,Dictionary<String,Object>)` | Creates a loss function by parsing the name string to a `LossType` enum value. |
| `GetDoubleParam(Dictionary<String,Object>,String,Double)` | Gets a double parameter from the dictionary with a fallback default value. |
| `GetIntParam(Dictionary<String,Object>,String,Int32)` | Gets an integer parameter from the dictionary with a fallback default value. |


---
title: "ParameterRegistry<T>"
description: "Manages named parameters for weight loading."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Manages named parameters for weight loading.

## For Beginners

This class is like a phone book for model parameters.
Each parameter has a name (like "encoder.conv1.weight") and we can look up
or set parameters by their names.

When loading pretrained weights, we need to know:

1. What parameters exist in our model
2. What shape each parameter should be
3. Where to actually put the weight data

This registry provides all three capabilities.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParameterRegistry` | Initializes a new empty parameter registry. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of registered parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNames` | Gets all registered parameter names. |
| `GetShape(String)` | Gets the expected shape for a parameter. |
| `Load(Dictionary<String,Tensor<>>,Func<String,String>,Boolean)` | Loads weights from a dictionary. |
| `Register(String,Int32[],Func<Tensor<>>,Action<Tensor<>>)` | Registers a parameter with getter and setter delegates. |
| `RegisterChild(String,ParameterRegistry<>)` | Registers a child ParameterRegistry with a prefix. |
| `RegisterLayer(String,ILayer<>)` | Registers a layer's weights and biases. |
| `RegisterLayers(String,IEnumerable<ValueTuple<String,ILayer<>>>)` | Registers multiple layers with a naming pattern. |
| `TryGet(String,Tensor<>)` | Tries to get a parameter by name. |
| `TrySet(String,Tensor<>)` | Sets a parameter by name. |
| `Validate(IEnumerable<String>,Func<String,String>)` | Validates weights against registered parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_parameters` | Maps parameter names to their accessors. |


---
title: "ParameterRange"
description: "Defines the range and type of a hyperparameter for AutoML search"
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Defines the range and type of a hyperparameter for AutoML search

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoricalValues` | List of possible values for categorical parameters |
| `DefaultValue` | Default value for the parameter |
| `MaxValue` | The maximum value for numeric parameters |
| `MinValue` | The minimum value for numeric parameters |
| `Step` | The step size for discrete parameters |
| `Type` | The type of parameter (Integer, Float, Boolean, Categorical, etc.) |
| `UseLogScale` | Whether to use logarithmic scale for sampling |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the ParameterRange, including deep cloning of reference-type properties |
| `DeepCloneList(List<Object>)` | Deep clones a list of objects, cloning each element if possible. |
| `DeepCloneObject(Object)` | Deep clones an object if possible, otherwise returns the object itself. |


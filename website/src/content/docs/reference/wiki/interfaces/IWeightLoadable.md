---
title: "IWeightLoadable<T>"
description: "Defines the contract for models that support loading weights by name."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for models that support loading weights by name.

## For Beginners

Think of this as a way to "transplant" knowledge from
pretrained models. Each weight has a name (like "encoder.conv1.weight") and
this interface lets us set those weights by their names.

Example:
```cs
// Load pretrained weights
var weights = safeTensorsLoader.Load("model.safetensors");

// Apply to model
if (model is IWeightLoadable<float> loadable)
{
loadable.SetParameter("encoder.conv1.weight", weights["encoder.conv1.weight"]);
}
```

## How It Works

This interface enables loading pretrained weights from external sources like
SafeTensors, HuggingFace, and ONNX files into AiDotNet models.

## Properties

| Property | Summary |
|:-----|:--------|
| `NamedParameterCount` | Gets the total number of named parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameterNames` | Gets all parameter names in this model. |
| `GetParameterShape(String)` | Gets the expected shape for a parameter. |
| `LoadWeights(Dictionary<String,Tensor<>>,Func<String,String>,Boolean)` | Loads weights from a dictionary of tensors using optional name mapping. |
| `SetParameter(String,Tensor<>)` | Sets a parameter tensor by name. |
| `TryGetParameter(String,Tensor<>)` | Tries to get a parameter tensor by name. |
| `ValidateWeights(IEnumerable<String>,Func<String,String>)` | Validates that a set of weight names can be loaded into this model. |


---
title: "ModelMetadata<T>"
description: "Represents metadata about a machine learning model, including its type, complexity, and additional descriptive information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents metadata about a machine learning model, including its type, complexity, and additional descriptive information.

## For Beginners

This class stores descriptive information about a machine learning model.

When working with machine learning models:

- You often need to know what type of model it is (linear regression, neural network, etc.)
- You want to understand its complexity and what features it uses
- You may need to store additional information about how it was created or should be used

This class stores all that information, including:

- The type of model (classification, regression, etc.)
- How many features (input variables) it uses
- How complex the model is
- A human-readable description
- Any additional custom information you want to include
- The actual model data in serialized form

This metadata helps you understand what a model does and how it works
without having to examine the model itself.

## How It Works

This class encapsulates metadata about a machine learning model, providing information that describes the model's 
characteristics without containing the actual model implementation. It includes details such as the model type, 
feature count, complexity, and a textual description. Additionally, it provides an extensible dictionary for storing 
arbitrary additional information and can store serialized model data. This metadata is useful for model cataloging, 
selection, and management purposes.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdditionalInfo` | Gets or sets additional information about the model as key-value pairs. |
| `Complexity` | Gets or sets a measure of the model's complexity. |
| `Description` | Gets or sets a human-readable description of the model. |
| `FeatureCount` | Gets or sets the number of features used by the model. |
| `FeatureImportance` | Gets or sets the importance of each feature in the model. |
| `ModelData` | Gets or sets the serialized model data. |
| `Name` | Gets or sets the name of the model. |
| `Properties` | Gets custom properties associated with the model. |
| `TrainingDate` | Gets or sets the date and time (with timezone) when the model was trained. |
| `Version` | Gets or sets the version of the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RemoveProperty(String)` | Removes a custom property from the Properties dictionary. |
| `SetProperty(String,Object)` | Adds or updates a custom property in the Properties dictionary. |


---
title: "HyperparameterSearchSpace"
description: "Defines the search space for hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Defines the search space for hyperparameter optimization.

## How It Works

**For Beginners:** A search space defines all possible values for each hyperparameter.
For example, learning rate might be between 0.001 and 0.1, and batch size might be 16, 32, or 64.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperparameterSearchSpace` | Initializes a new instance of the HyperparameterSearchSpace class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Parameters` | Gets or sets the parameter distributions/ranges. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBoolean(String)` | Adds a boolean parameter. |
| `AddCategorical(String,Object[])` | Adds a categorical parameter (discrete choices). |
| `AddContinuous(String,Double,Double,Boolean)` | Adds a continuous (real-valued) parameter. |
| `AddInteger(String,Int32,Int32,Int32)` | Adds an integer parameter. |


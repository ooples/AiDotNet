---
title: "ModelEvaluationInput<T, TInput, TOutput>"
description: "Represents the input data required for evaluating a machine learning model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Inputs`

Represents the input data required for evaluating a machine learning model.

## For Beginners

This class acts as a container for all the information needed to evaluate a model.
It includes the model itself, the data to evaluate it with, and information about how the data is preprocessed.

## How It Works

- The Model property holds the actual machine learning model to be evaluated.
- The InputData property contains the data used for evaluation, including inputs and expected outputs.
- The PreprocessingInfo property holds information about how the data has been transformed, which is important for

interpreting the results correctly.

## Properties

| Property | Summary |
|:-----|:--------|
| `InputData` | Gets or sets the input data used for model evaluation. |
| `Model` | Gets or sets the machine learning model to be evaluated. |
| `PredictionTypeOverride` | Gets or sets an optional override for the prediction type used when calculating metrics. |
| `PreprocessingInfo` | Gets or sets the preprocessing information for the input data. |


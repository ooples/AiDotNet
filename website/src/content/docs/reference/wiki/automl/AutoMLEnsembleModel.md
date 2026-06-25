---
title: "AutoMLEnsembleModel<T>"
description: "A simple tabular ensemble model used as a facade-safe AutoML final model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

A simple tabular ensemble model used as a facade-safe AutoML final model.

## For Beginners

Instead of trusting one model, an ensemble uses multiple models and combines their answers.
This often improves stability and accuracy.

## How It Works

This ensemble combines multiple `IFullModel` members by averaging (regression/binary)
or voting (multi-class) over their predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoMLEnsembleModel` | Initializes a new instance of the `AutoMLEnsembleModel` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Members` | Gets the member models in the ensemble. |
| `PredictionType` | Gets or sets the prediction type used to combine outputs (regression vs classification). |
| `Weights` | Gets or sets the per-member weights used when combining predictions. |


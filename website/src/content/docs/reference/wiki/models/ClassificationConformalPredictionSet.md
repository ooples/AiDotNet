---
title: "ClassificationConformalPredictionSet"
description: "Represents a conformal prediction set for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents a conformal prediction set for classification tasks.

## For Beginners

Instead of returning a single class label, conformal prediction can return a set of
possible classes. When the model is uncertain, the set tends to be larger. When the model is confident, the set
often contains a single class.

## How It Works

A conformal prediction set contains the class indices that are guaranteed (under standard conformal assumptions)
to contain the true class with the configured confidence level.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClassificationConformalPredictionSet(Int32[][])` | Initializes a new instance of the `ClassificationConformalPredictionSet` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassIndices` | Gets the predicted class index sets per sample. |


---
title: "IClassifier<T>"
description: "Defines the common interface for all classification algorithms in the AiDotNet library."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the common interface for all classification algorithms in the AiDotNet library.

## For Beginners

Classification is about putting things into categories.

For example, classification can be used to:

- Predict whether an email is spam or not spam (binary classification)
- Identify handwritten digits (0-9) from images (multi-class classification)
- Determine which diseases a patient might have (multi-label classification)
- Rate customer satisfaction as Poor/Fair/Good/Excellent (ordinal classification)

Unlike regression algorithms (which predict numbers), classification algorithms predict
which category or categories something belongs to.

## How It Works

Classification is a type of machine learning algorithm used to predict categorical values
(discrete classes) rather than continuous values. This interface extends IFullModel with
classification-specific functionality.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassLabels` | Gets the class labels learned during training. |
| `NumClasses` | Gets the number of classes that this classifier can predict. |
| `TaskType` | Gets the type of classification task this classifier is configured for. |


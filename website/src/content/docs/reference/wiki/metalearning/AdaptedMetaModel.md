---
title: "AdaptedMetaModel<T, TInput, TOutput>"
description: "Generic adapted model wrapper for meta-learning algorithms that use gradient-based inner-loop adaptation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Generic adapted model wrapper for meta-learning algorithms that use gradient-based inner-loop adaptation.
After adaptation, the model uses the adapted parameters for prediction.

## For Beginners

An adapted meta-model is the result of a meta-learning
algorithm (like MAML) adapting to a new task. After seeing a few examples of a new task,
the meta-learner produces this adapted model with task-specific parameters. Think of it
like a student who has learned general problem-solving skills and then quickly adapts
to a specific exam topic after seeing just a few practice questions.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptedSupportFeatures` |  |
| `ParameterModulationFactors` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `GetParameters` |  |
| `Predict()` |  |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |


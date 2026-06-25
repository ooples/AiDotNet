---
title: "ValidationCurveEngine<T>"
description: "Engine for generating validation curves: how model performance changes with hyperparameter values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Engine for generating validation curves: how model performance changes with hyperparameter values.

## For Beginners

Validation curves show how a hyperparameter affects performance:

- X-axis: Hyperparameter values (e.g., regularization strength)
- Y-axis: Performance metric (training and validation scores)
- Helps identify optimal hyperparameter range

## How It Works

**What validation curves tell you:**

- **Underfitting region**: Both train and val scores are low
- **Optimal region**: Val score peaks, train is slightly higher
- **Overfitting region**: Train high, val declining

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ValidationCurveEngine(ValidationCurveOptions)` | Initializes the validation curve engine. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Generate([0:,0:],[],Func<[0:,0:],[],Double,>,Func<,[0:,0:],[]>,String,Double[],String,Int32,Boolean,Boolean)` | Generates a validation curve for a given hyperparameter. |


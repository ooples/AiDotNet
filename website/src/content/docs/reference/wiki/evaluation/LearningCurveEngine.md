---
title: "LearningCurveEngine<T>"
description: "Engine for generating learning curves: how model performance changes with training set size."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Engine for generating learning curves: how model performance changes with training set size.

## For Beginners

Learning curves show how your model improves as you give it more data:

- X-axis: Number of training samples
- Y-axis: Performance metric (e.g., accuracy)
- Typically shows both training and validation scores

## How It Works

**What learning curves tell you:**

- **High bias (underfitting)**: Both curves plateau at low performance
- **High variance (overfitting)**: Large gap between train (high) and val (low)
- **Need more data**: Validation curve still improving at max training size
- **Good fit**: Both curves converge at high performance

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LearningCurveEngine(LearningCurveOptions)` | Initializes the learning curve engine. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Generate([0:,0:],[],Func<[0:,0:],[],>,Func<,[0:,0:],[]>,String,Double[],Int32,Boolean,Boolean)` | Generates a learning curve by training at various dataset sizes. |


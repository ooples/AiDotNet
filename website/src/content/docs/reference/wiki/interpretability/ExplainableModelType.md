---
title: "ExplainableModelType"
description: "Model types that the smart selector recognizes for choosing the optimal explainer."
section: "API Reference"
---

`Enums` · `AiDotNet.Interpretability`

Model types that the smart selector recognizes for choosing the optimal explainer.

## Fields

| Field | Summary |
|:-----|:--------|
| `BlackBox` | Any model with only predict access. |
| `KernelBased` | SVM, kernel regression. |
| `Linear` | Linear regression, logistic regression. |
| `NeuralNetwork` | Neural network with gradient access. |
| `TreeEnsemble` | Decision tree, random forest, gradient boosting. |


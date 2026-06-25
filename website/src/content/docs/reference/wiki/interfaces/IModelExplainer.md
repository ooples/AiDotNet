---
title: "IModelExplainer<T>"
description: "Interface for model-agnostic explainers that can explain any predictive model's decisions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for model-agnostic explainers that can explain any predictive model's decisions.

## For Beginners

Model explainers help you understand WHY a model makes certain predictions.
Unlike models that implement IInterpretableModel directly, these explainers work with ANY model -
they treat the model as a "black box" and analyze its behavior by observing inputs and outputs.

Think of it like understanding how a vending machine works: you don't need to see inside it,
you just try different button combinations and observe what comes out.

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` | Gets the name of this explanation method. |
| `SupportsGlobalExplanations` | Gets whether this explainer provides global (model-wide) explanations. |
| `SupportsLocalExplanations` | Gets whether this explainer provides local (per-instance) explanations. |


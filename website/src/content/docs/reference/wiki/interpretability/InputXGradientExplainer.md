---
title: "InputXGradientExplainer<T>"
description: "Input × Gradient attribution explainer - multiplies input values by their gradients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Input × Gradient attribution explainer - multiplies input values by their gradients.

## For Beginners

Input × Gradient is one of the simplest gradient-based attribution methods.
It multiplies each input feature by its gradient to get an attribution score.

**Intuition:**

- Gradient tells you: "If I change this feature, how much does the output change?"
- But gradient alone doesn't consider the feature's current value
- Input × Gradient says: "The attribution is both HOW MUCH the feature matters AND what its value is"

**Formula:**
attribution[i] = input[i] × gradient[i]

**Why multiply by input?**
Consider a feature x with gradient g:

- If x = 0 and g = 100: The feature COULD matter a lot, but currently contributes nothing
- If x = 10 and g = 0.1: The feature has high value but low sensitivity
- x × g captures both aspects

**Comparison to other methods:**

- Simpler than Integrated Gradients (just one gradient computation)
- Less theoretically grounded than SHAP (doesn't satisfy Shapley axioms)
- Good as a quick baseline or sanity check
- Can have issues with saturation (gradients near zero even for important features)

**When to use:**

- Quick initial analysis
- As a baseline to compare against more sophisticated methods
- When computational resources are limited
- For debugging (if Input×Gradient and SHAP disagree dramatically, investigate)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InputXGradientExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Int32,String[],Boolean)` | Initializes an Input × Gradient explainer with custom functions. |
| `InputXGradientExplainer(INeuralNetwork<>,Int32,String[],Boolean)` | Initializes an Input × Gradient explainer with a neural network. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Int32)` | Computes gradient of output w.r.t. |
| `Explain(Vector<>)` | Computes Input × Gradient attribution for an input. |
| `Explain(Vector<>,Nullable<Int32>)` | Computes Input × Gradient attribution for an input with a specific target class. |
| `ExplainBatch(Matrix<>)` |  |
| `GetPredictedClass(Vector<>)` | Gets the predicted class (argmax of output). |
| `GetPrediction(Vector<>)` | Gets the model prediction for an input. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |


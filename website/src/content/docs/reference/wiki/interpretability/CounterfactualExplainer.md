---
title: "CounterfactualExplainer<T>"
description: "Model-agnostic Counterfactual explainer that finds minimal changes needed for a different prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic Counterfactual explainer that finds minimal changes needed for a different prediction.

## For Beginners

Counterfactual explanations answer the question:
"What would need to change for the model to give a different prediction?"

Example: If a loan application was denied, a counterfactual might say:
"If your income was $5,000 higher, the loan would have been approved."

Key features:

- Shows the MINIMUM changes needed (fewest features changed)
- Changes should be realistic and actionable
- Helps users understand what they can do to get a different outcome

This is especially useful for:

- Credit decisions (what to improve for approval)
- Medical diagnoses (what tests would change the diagnosis)
- Any scenario where users want to know "what if"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CounterfactualExplainer(Func<Matrix<>,Vector<>>,Int32,Int32,Double,Int32,Double,String[],[],[],Boolean[],Nullable<Int32>)` | Initializes a new Counterfactual explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClipToRange(Double,Int32)` | Clips a value to the valid range for a feature. |
| `ComputeDistance(Vector<>,Vector<>)` | Computes the distance between original and counterfactual. |
| `CreateExplanation(Vector<>,Vector<>,Vector<>,Vector<>)` | Creates a CounterfactualExplanation from the results. |
| `Explain(Vector<>)` | Finds a counterfactual explanation for the given instance. |
| `Explain(Vector<>,Vector<>)` | Finds a counterfactual explanation to achieve a specific target prediction. |
| `ExplainBatch(Matrix<>)` |  |
| `SelectMutableFeatures(Random)` | Selects which features can be modified. |


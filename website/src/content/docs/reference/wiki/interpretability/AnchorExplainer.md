---
title: "AnchorExplainer<T>"
description: "Model-agnostic Anchor explainer that provides rule-based explanations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic Anchor explainer that provides rule-based explanations.

## For Beginners

Anchors are "if-then" rules that explain predictions.
Unlike SHAP or LIME which give feature weights, Anchors give you rules like:
"IF Age > 30 AND Income > 50000 THEN the model predicts 'Approved'"

Key concepts:

- **Precision**: How often the rule correctly predicts the same outcome (e.g., 95% of the time)
- **Coverage**: What fraction of all instances the rule applies to

Anchors are great when you need to explain to non-technical stakeholders
because rules are intuitive and easy to understand.

Example output: "The loan was approved because Age >= 25 AND CreditScore >= 700"
This rule holds for 95% of similar applicants (precision) and covers 30% of all applicants (coverage).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnchorExplainer(Func<Matrix<>,Vector<>>,Int32,Double,Int32,Int32,Int32,String[],[],[],Nullable<Int32>)` | Initializes a new Anchor explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateExplanation(HashSet<Int32>,Dictionary<Int32,ValueTuple<,>>,,)` | Creates an AnchorExplanation from the found anchor. |
| `CreateFeatureRule(Int32,,Random)` | Creates a feature rule (min, max) based on the instance value. |
| `EstimatePrecisionAndCoverage(Vector<>,Dictionary<Int32,ValueTuple<,>>,Int32,Random)` | Estimates precision and coverage of an anchor rule. |
| `Explain(Vector<>)` |  |
| `ExplainBatch(Matrix<>)` |  |
| `GenerateRandomSample(Vector<>,Dictionary<Int32,ValueTuple<,>>,Random)` | Generates a random sample, keeping anchor features fixed. |


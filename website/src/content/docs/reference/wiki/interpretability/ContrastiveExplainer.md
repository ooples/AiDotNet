---
title: "ContrastiveExplainer<T>"
description: "Contrastive explainer that answers \"Why X and not Y?\" questions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Contrastive explainer that answers "Why X and not Y?" questions.

## For Beginners

Contrastive explanations answer a specific type of question:
"Why did the model predict X instead of Y?"

This is how humans naturally ask for explanations:

- "Why was my loan denied (instead of approved)?"
- "Why is this classified as cat (and not dog)?"
- "Why is this patient high risk (rather than low risk)?"

Key concepts:

1. **Fact**: What actually happened (the model's prediction)
2. **Foil**: What you're comparing against (the alternative outcome)
3. **Pertinent Positives**: Features that support the fact
4. **Pertinent Negatives**: Features that, if changed, would lead to the foil

Why contrastive explanations are useful:

- Match how humans think about explanations
- Focus on what matters for the specific comparison
- Actionable: "Change these features to get the alternative outcome"

Example: "Loan denied instead of approved"

- Pertinent Positive: "Credit score is 580 (below 620 threshold)"
- Pertinent Negative: "If income were $10K higher, loan would be approved"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContrastiveExplainer(Func<Matrix<>,Vector<>>,Int32,String[],String[],[],[],Double)` | Initializes a new Contrastive explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFeatureContributions(Vector<>,Int32,Int32)` | Computes feature contributions to the fact-foil decision. |
| `ExpandToClassProbabilities(Vector<>,Int32)` | Expands a per-row prediction vector into a per-class probability vector. |
| `Explain(Vector<>)` | Explains why the model predicted one class instead of another. |
| `Explain(Vector<>,Int32,Int32)` | Explains why the model predicted factClass instead of foilClass. |
| `ExplainBatch(Matrix<>)` |  |
| `FindPertinentNegatives(Vector<>,Int32,Int32)` | Finds features that could flip the prediction (pertinent negatives). |
| `FindPertinentPositives(Vector<>,Int32,Int32)` | Finds features that support the fact (pertinent positives). |
| `GetPredictedClass(Vector<>)` | Gets the predicted class from scores. |
| `GetScoreDifference(Vector<>,Int32,Int32)` | Gets the score difference between fact and foil classes. |


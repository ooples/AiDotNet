---
title: "PrototypeExplainer<T>"
description: "Prototype-based explainer that explains predictions using similar examples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Prototype-based explainer that explains predictions using similar examples.

## For Beginners

Sometimes the best explanation is an example!
This explainer answers "Why this prediction?" with "Because it's similar to these examples."

How it works:

1. Maintain a set of "prototypes" (representative examples from training data)
2. For a new prediction, find the most similar prototypes
3. Explain the prediction based on these similar examples

Types of prototype explanations:

- **Nearest neighbors**: Show the K closest training examples
- **Same-class prototypes**: Show similar examples with the same prediction
- **Contrast prototypes**: Show similar examples with different predictions

Why prototypes are useful:

- Intuitive: "This loan was approved because it's similar to John's approved loan"
- Concrete: Shows actual examples, not abstract feature weights
- Trustworthy: Users can verify the similarity themselves

Example use cases:

- Medical diagnosis: "This case is similar to these past cases that were diagnosed as..."
- Credit decisions: "Your application is similar to these approved/denied applications"
- Image classification: "This image looks like these training images of cats"

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypeExplainer(Func<Matrix<>,Vector<>>,Matrix<>,Vector<>,Int32,DistanceMetric,String[],String[])` | Initializes a new Prototype explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistance(Vector<>,Vector<>)` | Computes distance between two vectors. |
| `Explain(Vector<>)` | Explains a prediction using similar prototypes. |
| `ExplainBatch(Matrix<>)` |  |


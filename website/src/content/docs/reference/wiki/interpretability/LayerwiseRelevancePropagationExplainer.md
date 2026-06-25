---
title: "LayerwiseRelevancePropagationExplainer<T>"
description: "Layer-wise Relevance Propagation (LRP) explainer for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Layer-wise Relevance Propagation (LRP) explainer for neural networks.

## For Beginners

LRP explains neural network predictions by propagating
"relevance" scores backward from the output to the input.

Key idea:

- Start with the prediction score as the total "relevance"
- At each layer, distribute relevance to neurons in the previous layer
- Continue until you reach the input features
- Each input feature gets a relevance score

Conservation principle:

- The total relevance is conserved at each layer
- Relevance in = Relevance out (like energy conservation)
- This means attributions sum to the prediction value

LRP rules (how to distribute relevance):

- **LRP-0 (Basic)**: Distribute proportional to contribution
- **LRP-ε (Epsilon)**: Adds stability with small epsilon
- **LRP-γ (Gamma)**: Emphasizes positive contributions
- **LRP-αβ**: Separately handles positive and negative contributions

When to use LRP:

- You want to understand which inputs were "responsible" for the output
- You need attributions that sum exactly to the prediction
- You want a principled way to handle different layer types

Example: For image classification:

- LRP shows which pixels contributed positively or negatively
- Red = positive relevance (supported the prediction)
- Blue = negative relevance (contradicted the prediction)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerwiseRelevancePropagationExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>[]>,Func<Int32,Matrix<>>,Int32,Int32[],String[],LRPRule,Double,Double)` | Initializes a new LRP explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeApproximateLRP(Vector<>,Int32)` | Computes approximate LRP using gradient × input (a simple approximation). |
| `ComputeFullLRP(Vector<>,Int32)` | Computes full LRP using layer activations and weights. |
| `ComputeNumericalGradient(Vector<>,Int32)` | Computes numerical gradient. |
| `Explain(Vector<>)` | Computes LRP relevance scores for an input. |
| `Explain(Vector<>,Int32)` | Computes LRP relevance scores for a specific output. |
| `ExplainBatch(Matrix<>)` |  |
| `PropagateAlphaBetaRule(Vector<>,Matrix<>,Double[],Double[],Double,Double)` | LRP-αβ rule separating positive and negative contributions. |
| `PropagateBasicRule(Vector<>,Matrix<>,Double[],Double[])` | LRP-0 Basic rule. |
| `PropagateEpsilonRule(Vector<>,Matrix<>,Double[],Double[])` | LRP-ε Epsilon rule for numerical stability. |
| `PropagateGammaRule(Vector<>,Matrix<>,Double[],Double[])` | LRP-γ Gamma rule emphasizing positive contributions. |


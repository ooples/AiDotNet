---
title: "AttentionVisualizationExplainer<T>"
description: "Attention Visualization explainer for Transformer models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Attention Visualization explainer for Transformer models.

## For Beginners

Transformers use "attention" to decide which parts of the input
to focus on when making predictions. This explainer visualizes these attention patterns.

What is attention?

- Each position in the input "attends" to all other positions
- Attention weights show how much each position influences another
- Higher weight = more influence/importance

Types of attention in Transformers:

1. **Self-attention**: Input attends to itself (e.g., words attending to other words)
2. **Cross-attention**: One sequence attends to another (e.g., decoder attending to encoder)

Visualization methods:

- **Raw attention**: Direct attention weights from the model
- **Attention rollout**: Combines attention across all layers
- **Attention flow**: Tracks information flow through the network

Example use cases:

- NLP: See which words the model focuses on for classification
- Vision Transformers: See which image patches are important
- Time Series: See which past timesteps influence predictions

Multi-head attention:

- Transformers have multiple "heads" that attend to different aspects
- This explainer can show individual heads or their combination

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AttentionVisualizationExplainer(Func<Tensor<>,Tensor<>>,Func<Tensor<>,Int32,Tensor<>>,Int32,Int32,Int32,String[])` | Initializes a new Attention Visualization explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAttentionRollout(List<[0:,0:]>)` | Computes attention rollout across all layers. |
| `ComputeAverageAcrossHeads([0:,0:,0:])` | Computes average attention across heads. |
| `ComputeTokenImportance([0:,0:])` | Computes overall token importance from attention rollout. |
| `Explain(Vector<>)` | Visualizes attention patterns for an input. |
| `ExplainBatch(Matrix<>)` |  |
| `ExplainTensor(Tensor<>)` | Visualizes attention patterns for an input tensor. |
| `ExtractAttentionMatrix(Tensor<>)` | Extracts attention matrix from tensor. |
| `GetAttentionFromPosition(AttentionExplanation<>,Int32,Int32,Int32)` | Gets attention for a specific token/position. |
| `GetTopAttendedPositions(AttentionExplanation<>,Int32,Int32,Int32)` | Gets the most attended positions for a query position. |


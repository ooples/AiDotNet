---
title: "AttentionExplanation<T>"
description: "Represents the result of an Attention Visualization analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of an Attention Visualization analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionRollout` | Gets or sets the attention rollout matrix [query, key]. |
| `AverageAttentionPerLayer` | Gets or sets average attention per layer (across heads) [layer][query, key]. |
| `LayerAttention` | Gets or sets attention weights for each layer [layer][head, query, key]. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of layers. |
| `SequenceLength` | Gets or sets the sequence length. |
| `TokenImportance` | Gets or sets the overall token importance (derived from rollout). |
| `TokenLabels` | Gets or sets labels for tokens/positions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedTokenImportance` | Gets tokens sorted by importance. |
| `ToString` | Returns a human-readable summary. |


---
title: "LinkPredictionEvaluation"
description: "Contains evaluation metrics for a link prediction model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Contains evaluation metrics for a link prediction model.

## For Beginners

These metrics measure how well the model predicts missing facts:

- MRR (Mean Reciprocal Rank): Average of 1/rank for each test triple. Higher = better. Perfect = 1.0.
- Hits@K: Fraction of test triples ranked in the top K. Higher = better. Perfect = 1.0.
- MeanRank: Average rank of correct answers. Lower = better. Perfect = 1.0.

## How It Works

Standard knowledge graph link prediction metrics in the filtered setting:
existing true triples are removed from the ranking before computing metrics.

## Properties

| Property | Summary |
|:-----|:--------|
| `HitsAtK` | Hits@K metrics: fraction of test triples where the correct entity is in the top K predictions. |
| `MeanRank` | Mean rank of the correct entity across all test triples. |
| `MeanReciprocalRank` | Mean Reciprocal Rank: average of 1/rank for correct entities. |
| `TestTripleCount` | Number of test triples evaluated. |


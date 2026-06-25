---
title: "TemporalTransEEmbedding<T>"
description: "Temporal TransE embedding: extends TransE with time-aware scoring via discretized time bins."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Temporal TransE embedding: extends TransE with time-aware scoring via discretized time bins.

## For Beginners

Regular TransE doesn't know about time — it treats all facts as eternal.
TE-TransE adds a time dimension:

- Facts like "Obama PRESIDENT_OF USA" have a time window (2009-2017)
- The model learns that "Obama + president_of + time_2012" ≈ "USA"
- But "Obama + president_of + time_2020" should NOT point to "USA"

Time is grouped into bins (e.g., one per year). Each bin has its own learned vector.

## How It Works

TE-TransE models temporal knowledge graphs by adding a time embedding to the scoring function:
d(h, r, t, τ) = ||h + r + t_time(τ) - t||, where t_time(τ) is a learned time bin embedding.
Time is discretized into bins (e.g., yearly), and each bin learns its own embedding vector.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDistanceBased` |  |
| `NumTimeBins` | Gets or sets the number of time bins for discretization. |
| `ResolvedNumTimeBins` | Gets the actual number of time bins after training (resolved from `NumTimeBins` or options default). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ScoreTripleAtTime(String,String,String,DateTime)` | Scores a triple at a specific time point. |


---
title: "RankingMetrics<T>"
description: "Static helpers for evaluating the quality of a ranking, most notably Normalized Discounted Cumulative Gain (NDCG@k)."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Metrics`

Static helpers for evaluating the quality of a ranking, most notably Normalized Discounted
Cumulative Gain (NDCG@k).

## For Beginners

When a model ranks items (stocks, search results, recommendations), you
need a number that says "how good is this ordering?". NDCG is the most common such number.
It rewards putting the truly-best items near the top and discounts items further down the list
(a great item ranked #1 counts for more than the same item ranked #20). It is normalized so
that a perfect ranking scores 1.0 and a poor ranking scores closer to 0, which makes scores
comparable across groups of different sizes. The "@k" version only looks at the top k positions,
which is what you want when you will only act on the top of the list (e.g. only buy the top 10).

## How It Works

DCG@k = Σ_{r=1..k} (2^{gain_r} - 1) / log2(r + 1), where gain_r is the true relevance of the
item the model placed at rank r. NDCG@k = DCG@k / IDCG@k, where IDCG@k is the DCG of the ideal
ordering (sorting items by true relevance descending).

## Methods

| Method | Summary |
|:-----|:--------|
| `Dcg(Int32[],Double[],Int32,Boolean)` | Discounted Cumulative Gain over the first `cutoff` positions of an ordering. |
| `NdcgAtK(Vector<>,Vector<>,Int32,Boolean)` | Computes Normalized Discounted Cumulative Gain at cutoff k (NDCG@k) for a single ranking group. |
| `StableArgsortDescending(Double[])` | Returns indices that sort `values` in descending order, breaking ties by original index (so the result is deterministic). |


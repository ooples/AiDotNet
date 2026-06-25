---
title: "RobustAggregationOptions"
description: "Configuration options for robust aggregation strategies in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for robust aggregation strategies in federated learning.

## How It Works

Robust aggregation helps defend against outliers and Byzantine (malicious or faulty) clients.

**For Beginners:** If some clients send "bad" updates (because of bugs, corrupted data, or attacks),
robust aggregation tries to reduce their impact so the global model stays stable.

Common strategies:

- Trimmed mean: drops extreme values before averaging.
- Median: takes the middle value per parameter (resistant to outliers).
- Krum/Multi-Krum: chooses the most "central" client updates by distance.
- Bulyan: combines Multi-Krum selection with trimming for stronger robustness.

## Properties

| Property | Summary |
|:-----|:--------|
| `ByzantineClientCount` | Gets or sets the assumed number of Byzantine clients (f) for Krum/Multi-Krum/Bulyan. |
| `GeometricMedianEpsilon` | Gets or sets a small epsilon used to avoid division-by-zero in RFA (Weiszfeld) updates. |
| `GeometricMedianMaxIterations` | Gets or sets the maximum number of iterations for geometric-median/RFA aggregation. |
| `GeometricMedianTolerance` | Gets or sets the convergence tolerance for geometric-median/RFA aggregation. |
| `MultiKrumSelectionCount` | Gets or sets how many client updates (m) Multi-Krum keeps before averaging. |
| `TrimFraction` | Gets or sets the trimming fraction for trimmed-mean based aggregations (0.0 to < 0.5). |
| `UseClientWeightsWhenAveragingSelectedUpdates` | Gets or sets whether robust strategies should use clientWeights when averaging selected updates. |


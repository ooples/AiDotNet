---
title: "LeidenOptions"
description: "Configuration options for the Leiden community detection algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Communities`

Configuration options for the Leiden community detection algorithm.

## For Beginners

The Leiden algorithm finds groups (communities) of tightly connected nodes.

- Resolution: Higher values find smaller, more fine-grained communities. Lower values merge into larger groups.
- MaxIterations: How many times the algorithm refines the communities. More iterations = better quality but slower.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxIterations` | Maximum number of iterations for the Leiden algorithm. |
| `Resolution` | Resolution parameter controlling community granularity. |
| `Seed` | Random seed for reproducibility. |


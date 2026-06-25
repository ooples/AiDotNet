---
title: "ClusteredClientSelectionStrategy"
description: "Cluster-based client selection using simple k-means over per-client embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Selection`

Cluster-based client selection using simple k-means over per-client embeddings.

## How It Works

**For Beginners:** This strategy tries to pick clients from different "types" of behavior by clustering
clients into groups and sampling from each cluster.


---
title: "GraphGenerationType"
description: "Type of graph generation approach."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Type of graph generation approach.

## For Beginners

These are different strategies for generating new graphs:

- **VariationalAutoencoder**: Learns a latent space representation and generates by sampling
- **Autoregressive**: Generates graphs one node/edge at a time sequentially
- **OneShot**: Generates the entire graph structure in a single pass

## Fields

| Field | Summary |
|:-----|:--------|
| `Autoregressive` | Autoregressive generation - generates nodes/edges sequentially. |
| `OneShot` | One-shot generation - generates entire graph at once. |
| `VariationalAutoencoder` | Variational autoencoder approach - learns latent space for generation. |


---
title: "FederatedHeterogeneityCorrectionOptions"
description: "Configuration options for federated heterogeneity correction algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated heterogeneity correction algorithms.

## How It Works

**For Beginners:** In federated learning, clients often have different data and different compute speeds.
Heterogeneity correction methods help reduce "client drift" so the global model converges more reliably on non-IID data.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the heterogeneity correction algorithm. |
| `ClientLearningRate` | Gets or sets the client learning rate used by methods that need it (e.g., SCAFFOLD control variates). |
| `FedDynAlpha` | Gets or sets the FedDyn regularization strength (alpha). |


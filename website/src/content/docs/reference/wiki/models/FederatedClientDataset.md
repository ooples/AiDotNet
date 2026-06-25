---
title: "FederatedClientDataset<TInput, TOutput>"
description: "Represents a single client's local dataset for federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a single client's local dataset for federated learning.

## How It Works

This type is intentionally simple: it holds the client's local features and labels
plus the sample count used for weighting during aggregation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedClientDataset(,,Int32)` | Initializes a new instance of the `FederatedClientDataset` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Features` | Gets the client's local feature data. |
| `Labels` | Gets the client's local label/target data. |
| `SampleCount` | Gets the number of samples contained in this dataset. |


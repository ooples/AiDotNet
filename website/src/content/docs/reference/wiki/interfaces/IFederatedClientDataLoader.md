---
title: "IFederatedClientDataLoader<T, TInput, TOutput>"
description: "Represents a data loader that can provide per-client datasets for federated learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a data loader that can provide per-client datasets for federated learning.

## For Beginners

Think of this as a "normal data loader" that also knows how to give you
each client's local data separately, so federated learning can train realistically.

## How It Works

Federated learning uses many small datasets (one per client/device/organization) instead of a single centralized dataset.
This interface allows a data loader to expose those natural partitions while still supporting the standard
`IInputOutputDataLoader` facade for aggregated access.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientData` | Gets the per-client datasets used for federated learning simulation. |


---
title: "IHomomorphicEncryptionProvider<T>"
description: "Provides homomorphic encryption operations for federated learning aggregation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Provides homomorphic encryption operations for federated learning aggregation.

## How It Works

**For Beginners:** The provider hides cryptographic details (keys, ciphertexts, parameters) behind a simple interface.

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateEncryptedWeightedAverage(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>,Vector<>,IReadOnlyList<Int32>,HomomorphicEncryptionOptions)` | Aggregates selected parameter indices using homomorphic encryption to produce a weighted average. |
| `GetProviderName` | Gets the provider name. |


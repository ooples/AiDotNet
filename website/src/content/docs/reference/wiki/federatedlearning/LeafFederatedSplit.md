---
title: "LeafFederatedSplit<TInput, TOutput>"
description: "Represents a single LEAF split (train/test) as per-client datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Represents a single LEAF split (train/test) as per-client datasets.

## For Beginners

Instead of one big dataset for everyone, federated learning uses
many small datasets — one per device or organization. This class stores that "one dataset per client"
view of the data.

## How It Works

LEAF is a federated learning benchmark suite where each user corresponds to one client.
This type preserves that structure by storing one dataset per user.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeafFederatedSplit(IReadOnlyList<String>,IReadOnlyDictionary<String,FederatedClientDataset<,>>)` | Initializes a new instance of the `LeafFederatedSplit` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientCount` | Gets the number of clients/users in this split. |
| `UserData` | Gets the per-user datasets in this split. |
| `UserIds` | Gets the ordered list of user IDs in this split. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToClientIdDictionary(IReadOnlyDictionary<Int32,String>)` | Converts this split into the `Dictionary<int, FederatedClientDataset>` form used by trainers. |


---
title: "LeafSent140FederatedDatasetLoader"
description: "Loads the LEAF Sent140 benchmark JSON files into per-client datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Loads the LEAF Sent140 benchmark JSON files into per-client datasets.

## For Beginners

This loader reads LEAF JSON and returns one dataset per user so federated learning
simulations match the benchmark's per-user partitioning.

## How It Works

Sent140 is a federated sentiment classification benchmark derived from tweets. LEAF stores each sample as an array
of string fields (id, date, query, user, text) and a numeric label (0/1).

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadDatasetFromFiles(String,String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Sent140 train dataset and optional test dataset from files. |
| `LoadSplitFromFile(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Sent140 split (train/test) from a JSON file. |
| `LoadSplitFromJson(String,LeafFederatedDatasetLoadOptions)` | Loads a LEAF Sent140 split (train/test) from a JSON string. |

